#pragma once
#include "Dispatcher.h"
#include "EventCalculation.h"
#include "EventToxicity.h"
#include "Setting.h"
#include <functional>
#include <string>
#include <list>

struct EventConfig;

class IToxicityIndexGeneric : public Dispatcher<EventCalculationGroup *>::Listener,
							  public Dispatcher<EventConfig *>::Listener,
							  public Dispatcher<EventToxicity *>,
							  public SettingsContainer
{
public:
	virtual void OnEvent(Dispatcher<EventCalculationGroup *> *sender, EventCalculationGroup *e) = 0;
	virtual void OnEvent(Dispatcher<EventConfig *> *sender, EventConfig *e) = 0;
	virtual void SetSerialId(int id) = 0;
	virtual Dispatcher<EventConfig *>::Listener *GetMortalityConfig() = 0;
};

class ICalculation;

class Mortality : public Dispatcher<EventConfig *>::Listener,
				  public SettingsContainer
{
public:
	enum StatusEnum
	{
		ALIVE = 0,
		ERROR,
		DEAD
	};

private:
	struct Status
	{
		int64_t time;
		StatusEnum status = ALIVE;
		std::list<int> m_alive;
		int m_nbZero;
	};
	std::map<ICalculation *const, Status> m_alive;

	///Mortality time in number of timebin
	Setting<int> m_mortalityTime;
	///Not use for now
	//Setting<int> m_mortalityRebirthTime ;
	///Camera serial
	int m_id = 0;

public:
	Mortality();
	void UpdateAlive(ICalculation *const, bool on_error, int64_t time);
	bool IsAlive(ICalculation *const) const;
	double Ratio() const;
	virtual void OnEvent(Dispatcher<EventConfig *> *sender, EventConfig *e) override;
	void SetSerialId(int id) { m_id = id; }
	int GetSerialId() const { return m_id; }
};

class ToxicityIndexMeanDistance : public IToxicityIndexGeneric
{
	//Options
	Setting<double> m_distanceSumExclusion;
	Setting<double> m_distanceLargeExclusion;
	SettingFactor<int64_t, 1000000> m_meanIntegrationTime;

	//State Variable
	double m_sumMean;
	int m_meanCount;
	int64_t m_nextIntgrationTime;
	///Average by animal
	std::map<ICalculation *, std::vector<double>> m_meanValues;
	///Mortality management
	Mortality m_mortality;

public:
	ToxicityIndexMeanDistance();

	void OnEvent(Dispatcher<EventCalculationGroup *> *sender, EventCalculationGroup *e) override;
	void SetSerialId(int id) override { m_mortality.SetSerialId(id); }

	Dispatcher<EventConfig *>::Listener *GetMortalityConfig() override { return &m_mortality; }
	virtual void OnEvent(Dispatcher<EventConfig *> *sender, EventConfig *e);

protected:
	virtual double toxIndexCalculation(const std::vector<double> &values);
	static double percentile(std::vector<double> values, int percent);
};

/**
 * @brief toxicity index made by George Ruck
 * Modification of default toxindex to 0-100%
 */
class ToxicityIndexPercentage
	: public ToxicityIndexMeanDistance
{
	//Options
	///{'G':[70],'E':[70],'R':[80]}
	Setting<int> m_percentileBkgNoise;
	///{'G':[19],'E':[18],'R':[5]}
	Setting<double> m_factorBkgNoise;
	///Default {'G':[2000],'E':[1000],'R':[250]}
	Setting<int> m_cutoff50percent;
	///Default {'G':[3500],'E':[2500],'R':[450]}
	Setting<int> m_cutoff75percent;
	///Default {'G':[12000],'E':[10000],'R':[1200]}
	Setting<int> m_cutoff100percent;
	///{'G':3120,'E':1869,'R':406}
	Setting<int> m_offset_logfit;

public:
	ToxicityIndexPercentage();

protected:
	virtual double toxIndexCalculation(const std::vector<double> &values) override;
	double toxPercent(double ToxInd);
};