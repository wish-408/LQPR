# python3
# Please install OpenAI SDK first：`pip3 install openai`
from openai import OpenAI

client = OpenAI(api_key="sk-c7e36dd880fe47e1afb1a43de39b718b", base_url="https://api.deepseek.com")

while(1) :
    question = input("请输入对话：")
    if question == 'exit':
        break

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": question},
        ],
        stream=False
    )
    print("")
    print("回复：")
    print( response.choices[0].message.content or "", end="", flush=True)
    print("--------------------------------------------")
    print("")   
    
     
    
    # approchs = ["Gemma-27B","Deepseek-67B", "Llama-8B"]
    
    # for approch in approchs :
        
    #     save_path = result_dir + get_date_time() + '_' + approch + '.xlsx'
    #     row_data = [approch]
    #     for i in range(9):
    #         data = random.uniform(55, 65)
    #         row_data.append(data)
    #     write_to_excel(save_path, row_data)
    