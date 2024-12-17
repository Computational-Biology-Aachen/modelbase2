# modelbase2

*modelbase2* is a Python package designed to enable *mechanistic learning*, bridging the gap between *mechanistic modeling* and *machine learning*.
The package enables you to integrate ordinary differential equation (ODE) models with data-driven techniques.
This combination allows for more accurate and interpretable predictions in systems where both physical laws and data-driven insights are valuable.
*modelbase2* thus facilitates the development of models that are not only data-efficient but also robust and capable of capturing complex system dynamics.
**Choose one of the notebooks below to start your journey!**

## Building and simulating models

In this first notebook you will learn how to build ODE models and do basic simulations with them.
This will allow you to create time courses and do steady-state analysis as shown below.

<div>
    <img src="assets/time-course.png"
         style="vertical-align:middle; max-height: 175px;" />
    <img src="assets/protocol-time-course.png"
         style="vertical-align:middle; max-height: 175px;" />
    <img src="assets/steady-state.png"
         style="vertical-align:middle; max-height: 175px;" />
</div>

[Start learning](basics.ipynb){ .md-button }

## Parameter scans

Parameter scans allow you to systematically assess the behaviour of your model dependent on the value of one or more parameters.
*modelbase2* has rountines to scan over, and easily visualise **time courses**, **protocol time courses**, and **steady states** for one or more parameters.
<div>
    <img src="assets/time-course-by-parameter.png"
         style="vertical-align:middle; max-height: 175px;" />
    <img src="assets/parameter-scan-2d.png"
         style="vertical-align:middle; max-height: 175px;" />
</div>

[Start learning](scans.ipynb){ .md-button }

## Metabolic control analysis

Metabolic control analysis answers the question: **what happens to the concentrations and fluxes if I slightly perturb the system?**
It is thus a *local* measurement about which reactions hold the most control.
If you ever read about **rate-limiting steps**, then this is for you!
<div>
    <img src="assets/variable-elasticity.png"
         style="vertical-align:middle; max-height: 175px;" />
    <img src="assets/parameter-elasticity.png"
         style="vertical-align:middle; max-height: 175px;" />
    <img src="assets/response-coefficient.png"
         style="vertical-align:middle; max-height: 175px;" />
</div>

[Start learning](mca.ipynb){ .md-button }

## Fitting

Almost every model at some point needs to be fitted to experimental data to be **validated**.
*modelbase2* offers highly customisable routines for fitting either **time series** or **steady-states**.

<img src="assets/fitting.png" style="max-height: 175px;" />

[Start learning](fitting.ipynb){ .md-button }


## Monte Carlo methods

Almost every parameter in biology is better described with a distribution than a single value.
Monte-carlo methods allow you to capture the **range of possible behaviour** your model can exhibit.
This is especially useful when you want to understand the **uncertainty** in your model's predictions.
*modelbase2* offers these Monte Carlo methods for all *scans*  ...
<div>
    <img src="assets/time-course.png"
         style='vertical-align:middle; max-height: 175px;'/>
    <span style='padding: 0 1rem; font-size: 2rem'>+</span>
    <img src="assets/parameter-distribution.png"
         style='vertical-align:middle; max-height: 175px'/>
    <span style='padding: 0 1rem; font-size: 2rem'>=</span>
    <img src="assets/mc-time-course.png"
         style='vertical-align:middle; max-height: 175px'/>
</div>
and even for *metabolic control analysis*
<div>
    <img src="assets/parameter-elasticity.png"
         style='vertical-align:middle; max-height: 175px;'/>
    <span style='padding: 0 1rem; font-size: 2rem'>+</span>
    <img src="assets/parameter-distribution.png"
         style='vertical-align:middle; max-height: 175px'/>
    <span style='padding: 0 1rem; font-size: 2rem'>=</span>
    <img src="assets/violins.png"
         style='vertical-align:middle; max-height: 175px'/>
</div>

[Start learning](monte-carlo.ipynb){ .md-button }

## Label models

Labelled models allow explicitly mapping the transitions between isotopomers variables.

<img src="assets/cbb-labeled.png" style="max-width: 30rem;">


[Start learning](label-models.ipynb){ .md-button }

## Mechanistic Learning

Mechanistic learning is the intersection of mechanistic modelling and machine learning.
*modelbase2* currently supports two such approaches: surrogates and neural posterior estimation.
**Surrogate models** replace whole parts of a mechanistic model (or even the entire model) with machine learning models.

<img src="assets/surrogate.png" style="max-height: 300px;">

This allows combining together multiple models of arbitrary size, without having to worry about the internal state of each model.
They are especially useful for improving the description of *boundary effects*, e.g. a dynamic description of downstream consumption.
**Neural posterior estimation** answers the question: **what parameters could have generated the data I measured?**
Here you use an ODE model and prior knowledge about the parameters of interest to create *synthetic data*.
You then use the generated synthetic data as the *features* and the input parameters as the *targets* to train an *inverse problem*.
Once that training is successful, the neural network can now predict the input parameters for real world data.

<img src="assets/npe.png" style="max-height: 175px;">


[Start learning](mxl.ipynb){ .md-button }