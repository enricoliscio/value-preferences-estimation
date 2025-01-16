# Value Preferences Estimation and Disambiguation
This is the code for the paper [Value Preferences Estimation and Disambiguation in Hybrid Participatory Systems](https://arxiv.org/abs/2402.16751).

Estimate participants' value profiles based on their choices and motivations in a PVE survey. Predict the values underlying participants' motivation through an NLP model. Use Active Learning to simulate iterative data label collection guided by the value profiles that most differ from the average.

- Install the requirements listed in `requirements.txt`.
- Contact the corresponding author (e.liscio@tudelft.nl) to get access to the survey data.
- Copy the `data` folder inside `value_profiles`.
- Example command: `python active_learning.py --verbose True --iterations 5 --warm-start 291 --sample-size 146 --strategy uncertainty`