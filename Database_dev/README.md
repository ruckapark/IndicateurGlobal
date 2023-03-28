# File explanation for database development

## Video compression

Finalised working version depends on ImageText module. In this code this is located on admin PC, but code has been added to folder in zip form for necessary use
Currently only run on Admin PC as relies on PythonScripts\Imagetext for all video cropping features

*It was necessary for data storage and irrelevance of certain data, to crop videos to 48 hours after the contaminant exposure*
**Excute successful - 1.5To of storage freed**

- **list_TxMfiles** : File written in 'I:\\' (NAS) to list all ToxMate files relevant to ToxPrints
- **db_dev1** : File to recreate dope reg with necessary columns for file location and thus compression
- **test_crop** : Single use of the crop function

**toxprint_cropvids** - the big bertha (crop all videos on the database to 48h post exposure)

## Creation of the database

*This relies on a pc with direct access to the NAS*

