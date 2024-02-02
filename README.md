Giles Story London 2024

The material in this repository is made available under the CC BY-NC 4.0 License (https://creativecommons.org/licenses/by-nc/4.0/), meaning you may use, modify, and distribute this work for non-commercial purposes only, citing the source as below:

"Story, G. W., Ereira, S., Valle, S., Chamberlain, S. R., Grant, J. E., Dolan, R. J. (2024) A computational signature of self-other mergence in Borderline Personality Disorder" 

Contains functions to fit models to probabilistic false belief task data as described 
in the above paper.

FBT_runmodels.m: script that loops through different model configurations to fit, this calls FBT_fit.m

FBT_fit.m: function to fit model to probabilistic false belief task data, with
options for ML, MAP or mixed effects optimisation; calls emfit.m; uses parfor to run different subjects in parallel

FBT_config.m: Function used to configure models and preprocess data, called by FBT_fit.m

FBT_llfun.m: Main model objective function 
