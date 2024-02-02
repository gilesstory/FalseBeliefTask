Giles Story London 2024

Contains functions to fit models to probabilistic false belief task data as described 
in "Story et al. (2024) A computational signature of self-other mergence in Borderline Personality Disorder"; 
Authors: Giles W. Story, Sam Ereira, Stephanie Valle, Samuel R. Chamberlain, Jon E. Grant, Raymond J. Dolan

FBT_runmodels.m: script that loops through different model configurations to fit, this calls FBT_fit.m

FBT_fit.m: function to fit model to probabilistic false belief task data, with
options for ML, MAP or mixed effects optimisation; calls emfit.m; uses parfor to run different subjects in parallel

FBT_config.m: Function used to configure models and preprocess data, called by FBT_fit.m

FBT_llfun.m: Main model objective function 
