# Instructions for recreating percentage

1. l'idee est de prendre une fonction sigmoid pour recreer la quantile mais en prenant en compte le systeme entier.

## Code

* data wrangling (functions utilise du fichier dataread_terrain)
* data_alive calcul - array (n*m) = (timebins * 16). 
* values == dead -> np.NaN  (ou on detecte un mort pour tous les timebins , remplace avec np.nan)
* IGT - create empty vector
* IGT - old create empty vector

* loop time bin
	* find coefficients
	* sum( values ** coefficients )
	* sum -> percent

* plot to compare