SEARCH_STRUCURE=../data/3SQG_C.pdb

#curl -X POST -F q='../data/3SQG_C.pdb' -F 'mode=3diaa' -F 'database[]=afdb50' -F 'database[]=afdb-swissprot' -F 'database[]=afdb-proteome' -F 'database[]=mgnify_esm30' -F 'database[]=pdb100' -F 'database[]=gmgcl_id' https://search.foldseek.com/api/ticket

#Parse
MMCIFDIR=../data/Foldseek_results/mmcif/
#mkdir $MMCIFDIR

#Download pdb files
#bash ./process/batch_download.sh -f $MMCIFDIR/ids.txt -c -o $MMCIFDIR
#Unzip
#gunzip $MMCIFDIR/*.gz
#Parse the mmcif files towards the interacting chains
mkdir $MMCIFDIR/seeds/
