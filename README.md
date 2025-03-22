# Abstract
Elicited performance requirements need to be quantified for compliance in different engineering tasks, e.g., configuration tuning and performance testing. Much existing work has relied on manual quantification, which is expensive and error-prone due to the imprecision. In this paper, we present `LQPR`, a highly efficient automatic approach for performance requirements quantification. `LQPR` relies on a new theoretical framework that converts quantification as a classification problem. Despite the prevalent applications of Large Language Models (LLMs) for requirement analytics, `LQPR` takes a different perspective to address the classification: we observed that performance requirements can exhibit strong patterns and are often short/concise, therefore we design a lightweight linguistically induced matching mechanism. We compare `LQPR` against nine state-of-the-art learning-based approaches over diverse datasets, demonstrating that it is ranked as the sole best for 88\% or more cases with two orders less cost. Our work proves that, at least for performance requirement quantification, specialized methods can be more suitable than the general LLM-driven approaches.

# Running Environment
```
python                        3.11.2
sklearn                       0.0.post5
transformers                  4.47.1
torch                         2.1.0
together                      1.2.7
spacy                         3.7.5
numpy                         1.24.3
pandas                        2.1.1
joblib                        1.2.0
```

# Program Entry Point
The program entry point is located in `code\main.py` at the root directory of the repository. To run the program, execute `python main.py` in the `code` folder.

The available operations are as follows:
```
if __name__ == '__main__':
    # bert_train()
    # RQ1()
    # RQ2()
    # RQ3()
    # RQ4()
    # example()
    pass
```

## Notes
### Fine-tuning BERT Model
In `code\main.py`, you can train the BERT model through the `bert_train()` function. Due to the upload limit of the git repository, we have not uploaded the 30 trained `bert` models. Instead, we have encapsulated the script for training the models. You can directly call this function to train the model to reproduce the experimental results.
### Obtaining the API Key of the Large Language Model
For the method of using a large language model for prediction, an `api-key` is required. Since the author cannot disclose the `api-key` they used, if you need to use it, please go to https://www.together.ai/ to obtain your private `api-key`, and set the environment variable `TOGETHER_API_KEY` of your system to the `api-key` you obtained. This `api-key` will be used in `code\LLM_TALK.py`:
```python
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
```

## Executing RQ1
In `code\main.py`, you can execute the experiment of RQ1 through the `RQ1()` function. This function will test the effects of various solutions on the `Promise` dataset that has been randomly divided 30 times, and save the results of `w_p`, `w_r`, and `w_f` in `expriment_result\Promise\RQ1`.

## Executing RQ2
In `code\main.py`, you can execute the experiment of RQ2 through the `RQ2()` function. This function will test the effects of various solutions on the three datasets of `LLM-GEN`, `PURE`, and `Shaukat_et_al`, and save the results of `w_p`, `w_r`, and `w_f` in `expriment_result\@dataset_name\RQ2`.

## Executing RQ3
`RQ2` is an ablation experiment. We will pass different configurations to `LQPR` and test their classification effects on 4 datasets. The experimental results will be saved in `expriment_result\@dataset_name\RQ3`. An example of the `LQPR` configuration is as follows:
```python
config = {
    'use_inv' : True,
    'use_sync' : True,
    'use_semantic' : True,
    'weight' : 0.7
}
```
* `use_inv` indicates whether the semantic inversion operation is required.
* `use_sync` indicates whether the longest common subsequence matching score is required.
* `use_semantic` indicates whether the semantic similarity score is required.
* `weight` represents the weight of the longest common subsequence matching score (the weight of the semantic similarity score is $1 - weight$).

## Executing RQ4
The code for testing the running time of each solution has been encapsulated in `RQ4()`. Executing this function will print the time taken by each method to process 89 requirement statements from the `Promise` dataset. For obtaining the memory overhead of each method, we use the `profile` annotation tool in the `memory_profiler` library, which will print out the memory overhead of each execution step throughout the whole process during runtime. You can obtain it by removing the comment symbol at the beginning of each method. For example, in `code\LQPR.py`:
```python
@profile
def LQPR(test_data_path, result_dir, save_name, config):
    ...
```

## Usage Example of LQPR
You can import the `predicte` method of `LQPR` in any function. This method takes a natural language sequence and a configuration dictionary as inputs. Note that before calling this function, you also need to execute the `init_pattern_vecs()` function first to load all the `patterns` and their word vector representations. The usage example is as follows:
```python
from LQPR import init_pattern_vecs, predicte

config = {
    'use_inv' : True,
    'use_sync' : True,
    'use_semantic' : True,
    'weight' : 0.7
}
init_pattern_vecs()
sentence = "The time taken to add products to the shopping cart must not exceed 2 milliseconds."
print(predicte(sentence, config))
``` 
The return result of the function is as follows:
```
('0 -1', ' exceed 100', 1.0, 'must not')
# (label, matched pattern, matching score, negative word)
```
