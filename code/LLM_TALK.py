import os
from together import Together
from static_data import *
from memory_profiler import profile
from tools import get_date_time, load_data, result_report

# @profile
def LLM_TALK(test_data_path, result_dir, save_name):

    models = ['google/gemma-2-27b-it', 'deepseek-ai/deepseek-llm-67b-chat', 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo']
    
    for i in range(3):

        model_name = models[i]
        print(model_name)
        
        sentences, real_labels = load_data(test_data_path)

        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

        question_path = '../prompt/question.txt'
        question = ''
        with open(question_path,'r',encoding='utf-8') as f:
            for line in f:
                question = question + line

        cnt = 0
        i = 0
        predicte_labels = []
        for sentence in sentences:
            
            stream = client.chat.completions.create(
                model = model_name,
                messages=[{"role": "user", 
                        "content": question + '\n' + sentence}],
                stream=True,
            )
            predicte_label = ''
            for chunk in stream:
                predicte_label  = predicte_label + chunk.choices[0].delta.content
        
            
            sign = 0
            if real_labels[i] in predicte_label:
                predicte_label = real_labels[i]
                
            else :

                for label in label_encode:
                    if label in predicte_label :
                        predicte_label = label
                        sign = 1
                        break    
        
            if sign:
                predicte_labels.append(predicte_label)
            else :
                predicte_labels.append("0 -1")
            
            i += 1            
            
        save_path = result_dir + get_date_time() + '_' + save_name + '_' +save_name + '_' + model_name.split('/')[0] + '.xlsx'
        result_report(model_name, save_path, predicte_labels, real_labels)
    
