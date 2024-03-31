# Stochastic Systems

### Setup

```
conda create --name stochastic-systems python=3.10

conda activate stochastic-systems

pip install swig
pip install -r requirements.txt
```

### TODO

1. Memory should be independent of agents - we can use same data to train many agents independently.
2. Add basic population mechanisms (population, selection, mutation):

   - Train 100 networks (RL)
   - Choose 10 best. (selection)
   - Random Gaussian step of weights (mutation) into 100 networks again.
   - Rinse and repeat

3. Add network architecture mutation. Test if it trains correctly.
4. Add CMAES.
5. Watch this spider walk
