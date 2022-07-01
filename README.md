# Training Camp 2022



#  References:

- Monthly energy consumption forecast: A deep learning approach
  https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Energy_consumption_in_households
    
https://ieeexplore.ieee.org/document/7966398
    

- Front. Energy Res., 22 October 2021 |
    https://doi.org/10.3389/fenrg.2021.779587
    
- Power Consumption Predicting and Anomaly Detection Based on Transformer and K-Means

Italian Household Load Profiles: A Monitoring Campaign
https://www.mdpi.com/2075-5309/10/12/217


https://www.sciencedirect.com/science/article/pii/S0169023X17303282

Learning process modeling phases from
modeling interactions and eye tracking data

https://www.techedgegroup.com/blog/data-science-process-problem-statement-definition


Luce

https://energy.poste.it/ui/index?accepted=L&L=211118LFIXPOSTE&commodity=L

Gas

https://energy.poste.it/ui/index?accepted=G&G=211118GFIXPOSTE&commodity=G


Terna

https://www.terna.it/en/electric-system/statistical-data-forecast/statistical-forecasts



### Predictive model
Trainable system fed by the fields parsed by the OCR (json file)

The output is the collection of forecasts for the next n months (n=6)

### Prescriptive model
Deterministic (non-trainable) model chosing the best commercial offer
on the base of the energy consumptions estimated by the predictive model

It is supervised by the predictive model

The commercial offer must fit the energy consumptions estimates
and the exceeding of the energy threshold forecasted by
the predictive model.

Steps
* Generate a dataset with at least 100k records
 
  The data range must cover at least 6 months 
  
  The dataset must be compliant with respect to the
  fields of the json file produced by the OCR
  (obtained by scanning energy bills from customers
  previous energy suppliers or prospect)

* Define a baseline model for the predictive model

* Define a baseline model for the prescriptive model

* Compare the baseline models with other simple models


# Training Camp 2022
## Title: Driving Business Decisions through AI

### Abstract
Algorithmic Business Thinking is a paradigm defining algorithms bases
on a symbiotic cooperation of humans and machines working side-by-side
in mitigating the risk of unconscious biases
that could cause effectiveness loss on the final decision.

The partnership of humans and machines derives AI driven business decisions
in a faster and more effective way, supporting better scaling
when heterogeneous business use-cases demanding increases.

AI algorithms will improve marketing strategies and
drive product evolutionary transformation,
embedding the AI in the product itself or using AI to design for innovation.
On the other hand, in some cases this paradigm may raise the need
for an ethical consilience among decisions provided by machine vs human
("Is this the right thing to do; what are the unintended consequences?")
This camp is aimed at the accomplishment of the following targets:
- Collect the data provided during the on-boarding of a customer
  purchasing energy supply offer in order
  to predict the energy consumption in the next period of time,
- Drive the identification of a business decision (i.e. commercial offer),
  and ensure consilience of sustainability and consumption demand.

As a prerequisite for this training, it's supposed that the students start
from a solid knowledge of Python and machine learning
most frequently adopted libraries.