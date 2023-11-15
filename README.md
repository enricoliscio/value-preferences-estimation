# al-value-profiles
Estimate participants' value profiles based on their choices and motivations in a PVE survey. Use Active Learning to simulate iterative data label collection guided by the value profiles that most differ from the average.

- Add the `data` folder inside `value_learning`.
- Run with something like: `python3 active_learning.py --verbose True --iterations 5 --warm-start 291 --sample-size 146 --strategy uncertainty`