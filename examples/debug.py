from vllm import LLM, SamplingParams

model = LLM(
            # 'meta-llama/Llama-2-7b-chat-hf',
            'meta-llama/Llama-3.1-8B-Instruct',
            download_dir='/data1/huggingface',
            gpu_memory_utilization=0.75,
            load_format='safetensors',
            kv_buffer_size=32,
            max_paddings=4096,
            max_num_batched_tokens=131072,
            dtype='float16',
            disable_log_stats=False,
            enforce_eager=True)

sampling_params = SamplingParams(temperature=0.0, max_tokens=512)  # greedy sampling

quant_config = [8, 4, 4, 2]
quant_groups = [1, 2, 2, 4]
compress_config = [0.0, 1.0]

# quant_config = [8, 8]
# quant_groups = [1, 1]
# compress_config = [0.0, 0.0]


prompts = [
    "Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , " \
    "commonly referred to as Valkyria Chronicles III outside Japan , " \
    "is a tactical role playing video game developed by Sega and Media.Vision for the PlayStation Portable ."
]

# prompts += [
#     "A robot may not injure a human being",
#     "To be or not to be,",
#     # "What is the meaning of life?",
#     # "It is only with the heart that one can see rightly",
#     # "How to recover from a breakup?",
#     # "Which American-born Sinclair won the Nobel Prize for Literature in 1930?",
# ]

prompts += [
    "The family later goes to the hospital to visit the farmer who was injured in the giant accident, only to find that Mr. Charming (revealed to be Prince Charming) has beaten them and erased the farmer's memory. Despite this, they interview the farmer's wife at his bedside. The farmer's wife, Mrs. Applebee, informs them that her husband had sworn he'd seen a giant, but she believes a different theory. She says there was a British man who often visited their farm and asked to rent their field, but became hostile when they refused. She says that later, the man had returned, apologized for being so rude, and offered to pay for them to stay in New York City as an apology. Mrs. Applebee had gone with her sister rather than her husband. When they arrived, the hotel had no record of their reservation. On the way out, the family is ambushed by a group of 'goons' who threaten the Grimms to abandon the case. Granny is not scared, and instead sees this as a sign they are on the right path. Granny decides to follow the gang and find out who employed them in a stakeout. On the way, she tells them about giants. the only person to ever have successfully robbed and killed a giant was Jack,(from Jack and the Beanstalk), but now he works at a retail store in town. On the stakeout, while granny and canis are distracted, sabrina makes an attempt to escape with Daphne, despite Daphne's protests. just after they leave the car, it is attacked by a giant. The giant, chanting about how he must find \"the englishman\" picks up the car, containing granny and canis, and walks away with it, leaving the girls alone in the woods with only granny's handbag. They try to hitchhike, but encounter Officer Hamstead, one of the three little pigs. He offers to drive the girls home, but they discover he works for Mr. Charming, and make an escape. The girls follow pixie lights into the woods and soon meet Puck (from A Midsummer Night's Dream). Puck originally believes they are spies and tries to drown them, claiming they have stolen the old lady away from him. they mistake  him for the infamous Peter Pan, which enrages him even further. He originally decides that he won't help them find granny because he is a self-proclaimed villain. Ultimately he follows them home, helps them get back into the house, and agrees to help them save their grandmother just because she was kind to him and fed him since he was little. Puck and Sabrina share a clear hatred for each other and spend the majority of the time bickering. Sabrina and Daphne find their father's diary, detailing his accounts with Mayor Charming. It reveals that the upcoming fundraiser ball at prince charming's mansion is a scam he created to make money after a series of business fails. They also find out that giants are very gullible. They theorize that Mayor Charming tricked a giant into crushing the house for him, and that Mayor Charming is the 'Englishman'."
]

outputs = model.generate(
    prompts, sampling_params, quant_config, quant_groups, compress_config)

print(outputs)
