#pragma once
#include "Dispatcher.h"
#include "EventCalculation.h"
#include "EventToxicity.h"
#include "Setting.h"
#include <functional>
#include <string>
#include <list>

struct EventConfig ;

class IToxicityIndexGeneric :
	public Dispatcher<EventCalculationGroup*>::Listener,
	public Dispatcher<EventConfig*>::Listener,
	public Dispatcher<EventToxicity*>,
	public SettingsContainer
{
public:
	virtual void OnEvent(Dispatcher<EventCalculationGroup*>* sender, EventCalculationGroup* e) = 0;
	virtual void OnEvent(Dispatcher<EventConfig*>* sender, EventConfig* e) = 0;
	virtual void SetSerialId(int id) = 0;
	virtual Dispatcher<EventConfig*>::Listener* GetMortalityConfig() = 0 ;
};

class ICalculation ;

class Mortality :
	public Dispatcher<EventConfig*>::Listener,
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
	struct Status {
		int64_t time ;
		StatusEnum status = ALIVE;
		std::list<int> m_alive;
		int m_nbZero;
	};
	std::map<ICalculation* const,Status> m_alive ;
	
	///Mortality time in number of timebin
	Setting<int> m_mortalityTime ;
	///Not use for now
	//Setting<int> m_mortalityRebirthTime ;	
	///Camera serial
	int m_id = 0;

public:
	Mortality();
	void UpdateAlive( ICalculation* const, bool on_error, int64_t time );
	bool IsAlive( ICalculation* const) const ;
	double Ratio() const ;
	virtual void OnEvent(Dispatcher<EventConfig*>* sender, EventConfig* e) override;
	void SetSerialId(int id) { m_id = id; }
	int GetSerialId() const { return m_id; }
};


class ToxicityIndexMeanDistance :
	public IToxicityIndexGeneric
{
//Options
	Setting<double> m_distanceSumExclusion ;
	Setting<double> m_distanceLargeExclusion ;
	SettingFactor<int64_t,1000000> m_meanIntegrationTime ;

//State Variable
	double m_sumMean ;
	int m_meanCount ;
	int64_t m_nextIntgrationTime ;
	///Average by animal
	std::map<ICalculation*, std::vector<double>> m_meanValues;
	///Mortality management 
	Mortality m_mortality ;
public:
	ToxicityIndexMeanDistance();

	void OnEvent(Dispatcher<EventCalculationGroup*>* sender, EventCalculationGroup* e) override ;
	void SetSerialId(int id) override { m_mortality.SetSerialId(id); }
	std::map<ICalculation*, std::vector<double>> getMeanValues(){return m_meanValues;}

	Dispatcher<EventConfig*>::Listener* GetMortalityConfig() override { return &m_mortality; }	
	virtual void OnEvent(Dispatcher<EventConfig*>* sender, EventConfig* e);
protected:
	static double percentile( std::vector<double> values, int percent );
};


class Percentage
	// public ToxicityIndexMeanDistance
	{
//Options
	Setting<double> p_percentileBDF ;
	Setting<double> p_factorBDF;
	Setting<int> p_cutoff50percent ;
	Setting<int> p_cutoff75percent ;
	Setting<int> p_cutoff100percent ;
	Setting<int> p_offset_logfit ;
	
	ToxicityIndexMeanDistance* toxicityIndexMeanDistance;
	
	
	public:
	Percentage(ToxicityIndexMeanDistance& p);

protected:
	double toxPercent( double ToxInd );
}