from rpy2.robjects.packages import importr
from rpy2 import robjects


# Set R library path
robjects.r('.libPaths("C:/Users/FX506/Documents/R_libs")')

# Load required R packages
base = importr("base")
libstable4u = importr("libstable4u")
stabledist = importr("stabledist")
cubature = importr("cubature")
alphastable = importr("alphastable")
invgamma = importr("invgamma")
LaplacesDemon = importr("LaplacesDemon")
lubridate = importr("lubridate")
magrittr = importr("magrittr")
mltest = importr("mltest")
evmix = importr("evmix")
nprobust = importr("nprobust")
BNPdensity = importr("BNPdensity")
stats = importr("stats")
