# http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
mkdir ./hyperspectral_datas
mkdir ./hyperspectral_datas/indian_pines
mkdir ./hyperspectral_datas/indian_pines/data
mkdir ./hyperspectral_datas/salina
mkdir ./hyperspectral_datas/salina/data

wget http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat -P ./hyperspectral_datas/indian_pines/data
wget http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat -P ./hyperspectral_datas/indian_pines/data

wget http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat -P ./hyperspectral_datas/salina/data
wget http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat -P ./hyperspectral_datas/salina/data
