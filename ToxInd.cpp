#include "stdafx.h"
#include "ToxicityIndex.h"
#include "Log.h"
#include "Exception.h"
#include "EventCalculationTrack.h"
#include "Config.h"
#include <algorithm>
#include <numeric>
#include <cmath>

Mortality::Mortality()
	: m_mortalityTime(this, "mortalityTime", 180) //180 = 180 * 20 timbin = 60 min
//	,m_mortalityRebirthTime(this,"mortalityRebirth",0) // 0 not use
{
}

void Mortality::UpdateAlive(ICalculation *const pCalculation, bool on_error, int64_t time)
{
	//LOG(LOG_DEBUG,"m_mortalityTime=%d,m_mortalityRebirthTime=%d", m_mortalityTime.get(), m_mortalityRebirthTime.get() );

	if (m_alive.find(pCalculation) == m_alive.end())
	{
		Status status;
		status.time = time;
		status.status = ALIVE;
		m_alive[pCalculation] = status;
	}
	auto &alive = m_alive[pCalculation];
	const size_t &length = alive.m_alive.size();

	const int previous = (length > 1) ? alive.m_alive.back() : 1;
	alive.m_alive.push_back(on_error ? 0 : 1); // Alive 1 else Dead 0
	if (alive.m_alive.size() > 15)			   //15 value = 3 x 5 min
		alive.m_alive.pop_front();

	if (alive.m_nbZero >= 0) //Alive
	{
		const int current = alive.m_alive.back();
		std::list<int>::iterator it = alive.m_alive.end();
		if (current && previous)
			alive.m_nbZero = 0;
		else
			alive.m_nbZero++;
	}
	else
	{ // Dead
		auto min_value = std::min_element(alive.m_alive.begin(), alive.m_alive.end());
		if (*min_value != 0)
		{
			alive.m_nbZero = 0; //rebirth
			alive.status = ALIVE;
		}
	}

	if (alive.m_nbZero >= m_mortalityTime.get())
	{						 //180 = 180 * 20 timbin = 60 min
		alive.m_nbZero = -1; //dead
		alive.status = DEAD;
	}
}

bool Mortality::IsAlive(ICalculation *const pCalulation) const
{
	std::map<ICalculation *const, Status>::const_iterator it = m_alive.find(pCalulation);
	if (it == m_alive.end())
		return true;
	else
		return (it->second.status != DEAD);
}

double Mortality::Ratio() const
{
	int nb = std::count_if(m_alive.begin(), m_alive.end(), [this](std::pair<ICalculation *const, Status> p) { return this->IsAlive(p.first); });
	return (1.0 - (double)nb / m_alive.size());
}

void Mortality::OnEvent(Dispatcher<EventConfig *> *sender, EventConfig *e)
{
	LOG(LOG_INFO, "Mortality::OnEvent Configuration change");

	if (m_id != e->serial)
		return;

	Update(e->value);
}

/**
 * @brief Percentile calculcation (quantile) with interpolation
 * like R or numpy.percentile
 * @return percentile from values, nan if list is empty
 */
double ToxicityIndexMeanDistance::percentile(std::vector<double> values, int percent)
{
	if (values.size() == 0)
		return std::numeric_limits<double>::quiet_NaN();

	std::sort(values.begin(), values.end());
	double q = percent / 100.0;
	double index = q * (values.size() - 1);
	int i = int(index);
	if (i == index)
	{
		return values[i];
	}
	else
	{
		int j = i + 1;
		double left = j - index;
		double right = index - i;
		return values[i] * left + values[i + 1] * right;
	}
}

ToxicityIndexMeanDistance::ToxicityIndexMeanDistance()
	: m_distanceSumExclusion(this, "distanceSumExclusion", std::numeric_limits<double>::max()),
	  m_distanceLargeExclusion(this, "distanceLargeExclusion", std::numeric_limits<double>::max()),
	  m_meanIntegrationTime(this, "meanIntegrationTime", 2 * 60), //2min
	  m_sumMean(0),
	  m_meanCount(0),
	  m_nextIntgrationTime(0)
{
}

double ToxicityIndexMeanDistance::toxIndexCalculation(const std::vector<double> &values)
{
	const int n_percentile = 5;
	double toxIndex = ToxicityIndexMeanDistance::percentile(values, n_percentile);
	return toxIndex * toxIndex;
}

void ToxicityIndexMeanDistance::OnEvent(Dispatcher<EventCalculationGroup *> *sender, EventCalculationGroup *e)
{
	if (m_nextIntgrationTime == 0)
		m_nextIntgrationTime = e->GetTimestamp() + m_meanIntegrationTime.get();

	if (e->GetTimestamp() >= m_nextIntgrationTime)
	{
		std::vector<double> means;
		for (auto &item : m_meanValues)
		{
			double mean = std::accumulate(item.second.begin(), item.second.end(), 0.0) / item.second.size();
			LOG(LOG_DEBUG, "Add mean %f count %d", mean, item.second.size());
			means.push_back(mean);
		}
		double index = toxIndexCalculation(means);

		m_nextIntgrationTime = e->GetTimestamp() + m_meanIntegrationTime.get();

		EventToxicity new_event(*e, index, m_mortality.Ratio());
		Dispatch(&new_event);

		m_meanValues.clear();
	}

	double meanGoupSum = 0.0;
	size_t meanGroupCount = 0;

	for (auto &&item : e->GetValues())
	{
		auto ev = std::dynamic_pointer_cast<EventCalculationTrack>(item.second);
		const auto &values = ev->GetValues();

		double sum_distance = values.at(MS_INACTIVITY).distance + values.at(MS_SMALL).distance + values.at(MS_LARGE).distance;
		if (sum_distance > m_distanceSumExclusion.get())
			continue;
		if (values.at(MS_LARGE).distance > m_distanceLargeExclusion.get())
			continue;

		//TODO on error pb counter bug
		bool on_error = (sum_distance == 0.0);

		m_mortality.UpdateAlive(item.first, on_error, e->GetTimestamp());

		if (m_mortality.IsAlive(item.first) == false)
			continue; //Dead

		// if (on_error)
		// continue ; //In case of error not use this value

		LOG(LOG_DEBUG, "[%d] Add sum distance %f", ev->GetId(), sum_distance);
		m_meanValues[item.first].push_back(sum_distance);
	}
}

void ToxicityIndexMeanDistance::OnEvent(Dispatcher<EventConfig *> *sender, EventConfig *e)
{
	LOG(LOG_INFO, "ToxicityIndexMeanDistance::OnEvent Configuration change");

	if (m_mortality.GetSerialId() != e->serial)
		return;

	Update(e->value);
}

ToxicityIndexPercentage::ToxicityIndexPercentage()
	: m_percentileBkgNoise(this, "percentileBkgNoise", 70),
	  m_factorBkgNoise(this, "factorBkgNoise", 19),
	  m_cutoff50percent(this, "cutoff50percent", 2000),
	  m_cutoff75percent(this, "cutoff75percent", 3500),
	  m_cutoff100percent(this, "cutoff100percent", 12000),
	  m_offset_logfit(this, "offset_logfit", 3120)
{
}

/**
 * @brief retrieve no-ranged toxindex and apply gruck tox index percentage fit
 */
double ToxicityIndexPercentage::toxIndexCalculation(const std::vector<double> &values)
{
	//No data return nan
	if (values.size() == 0)
		return std::numeric_limits<double>::quiet_NaN();

	//No ranged tox index old version
	const double toxIdx = ToxicityIndexMeanDistance::toxIndexCalculation(values);

	//Background noise
	const double bkgNoise = ToxicityIndexMeanDistance::percentile(values, m_percentileBkgNoise.get()) / m_factorBkgNoise.get();

	const double log_offset = 20.0 / std::log((m_cutoff100percent.get() - m_offset_logfit.get()) / (m_cutoff75percent.get() - m_cutoff50percent.get()));

	if (toxIdx < m_cutoff50percent.get())
	{
		return bkgNoise + (toxIdx * 45.0) / m_cutoff50percent.get();
	}
	else if (toxIdx < m_cutoff75percent.get())
	{
		return bkgNoise + ((toxIdx - m_cutoff50percent.get()) * 25.0) / (m_cutoff75percent.get() - m_cutoff50percent.get()) + 45.0;
	}
	else
	{
		double v = bkgNoise + std::log((toxIdx - m_offset_logfit.get()) / (m_cutoff75percent.get() - m_offset_logfit.get())) * log_offset + 70.0;
		return std::min(100.0, v);
	}
}