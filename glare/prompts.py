def get_naive_rag_prompt1(fact, criminal, precedents, precedent_database):
    examples = precedents
    case_list = precedent_database
    texts = []
    for caseid in examples:
        case = case_list[caseid-1]
        cur_article_label = case["meta"]["relevant_articles"][0]
        cur_charge_label = case["meta"]["accusation"][0]
        text = "本案的相关法条是：" + f"第{cur_article_label}条\n" + "本案的被告人罪名是：" + f"{cur_charge_label}\n"
        texts.append(text)
    examples = "\n".join(texts)
    return f"你要做的任务是相关法条预测和罪名预测，用<>括起来的是本案事实，用---括起来的是与本案内容类似的5个案件以及相应的法条和罪名。请一步一步地思考，根据你自己的知识，并参考5个案件的相关法条、罪名，分析本案的相关法条和罪名预测结果。重点关注用()括起来的罪犯的犯罪行为。\n"+f"<"+fact+f">"+f"\n---"+examples +f"---\n"+f"("+criminal +f")\n"
    


def get_naive_rag_prompt2(fact,criminal,predict):
    return f"用<>括起来的是本案事实，用---括起来的是模型利用自己的知识对案例进行一步一步地分析得到的相关法条和罪名预测结果，请根据本案事实和预测结果，从中总结提取出最相关的一个法条和罪名,重点关注用()括起来的罪犯的犯罪行为。输出格式为：本案的相关法条是：第*条，本案的罪名是：*罪。\n"+f"<"+fact+f">"+f"\n---"+predict +f"---\n"+f"("+criminal +f")\n"

def get_naive_rag_prompt3(fact, criminal, precedents, precedent_database,output):
    examples = precedents
    case_list = precedent_database
    texts = []
    for caseid in examples:
        case = case_list[caseid-1]
        cur_fact = case["fact"].replace(" ", "")
        cur_article_label = case["meta"]["relevant_articles"][0]
        cur_charge_label = case["meta"]["accusation"][0]
        text = "本案的事实是：" + f"{cur_fact}\n" +"相关法条是：" + f"第{cur_article_label}条\n" + "被告人罪名是：" + f"{cur_charge_label}\n"
        #text = "本案的相关法条是：" + f"第{cur_article_label}条\n" + "本案的被告人罪名是：" + f"{cur_charge_label}\n"
        texts.append(text)
    examples = "\n".join(texts)
    if "没有相似法条" in output:
    #if output == '没有相似法条！':
        return f"你要做的任务是相关法条预测和罪名预测，用<>括起来的是本案事实，用---括起来的是与本案内容类似的5个案件以及相应的法条和罪名。请一步一步地思考，根据你自己的知识，并参考5个案件的相关法条、罪名，分析本案的相关法条和罪名预测结果。重点关注用()括起来的罪犯的犯罪行为。\n"+f"<"+fact+f">"+f"\n---"+examples +f"---\n"+f"("+criminal +f")\n"
    else:
        return f"你要做的任务是相关法条预测和罪名预测，用<>括起来的是本案事实，用---括起来的是与本案内容类似的5个案件以及相应的法条和罪名，用[]括起来的是对相似法条内容的理解和区分。请一步一步地思考，根据你自己的知识，并参考5个案件的相关法条、罪名，分析本案的相关法条和罪名预测结果。重点关注用()括起来的罪犯的犯罪行为。\n"+f"<"+fact+f">"+f"\n---"+examples +f"---\n"+f"["+output+f"]\n"+f"("+criminal +f")\n"

def get_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "你是一名推理助手，具有进行网络搜索的能力，你能够准确进行法律判决预测。你有特殊工具：\n\n"
        "- 进行搜索：写入 <|begin_search_query|> 你的查询内容 <|end_search_query|>。\n"
        "然后，系统将搜索并分析相关的网页，并以 <|begin_search_result|> ...搜索结果... <|end_search_result|> 的格式将有用信息返回给你。\n"
        f"可以多次重复搜索过程。搜索尝试的最大次数限制为 {MAX_SEARCH_LIMIT}。\n"
        "一旦你获得了所需的信息，继续你的推理。\n\n"
        "示例：\n"
        "问题：\"如何区分刑法贷款诈骗罪和骗取贷款罪\"\n"
        "Assistant思考步骤：\n"
        "- 我需要找出贷款诈骗罪和骗取贷款罪的主要区别。\n\n"
        "Assistant：\n"
        "<|begin_search_query|>区分贷款诈骗罪和骗取贷款罪<|end_search_query|>\n\n"
        "(系统返回搜索结果)\n\n"
        "Assistant：\n"
        "<|begin_search_result|> ...搜索结果... <|end_search_result|>\n\n"
        "现在Assistant已经有足够的信息，可以继续推理。\n\n"
        "注意：\n"
        "- 使用 <|begin_search_query|> 请求网络搜索，并以 <|end_search_query|> 结束。\n"
        "- 当完成信息检索后，继续你的推理。\n\n" 
    ) 


def get_instruction_add_reasoning_chain(MAX_SEARCH_LIMIT):
    return (
        "你是一名推理助手，具有调用外部工具的能力。你有特殊工具：\n"
        "- 扩展工具，对初步预测得到的候选罪名进行扩展：写入 <|begin_expand|> 罪名1, 罪名2, ...  <|end_expand|>。\n"
        "系统将使用已有的工具对你提供的罪名进行扩展，以获取更全面的候选罪名，同时还会给出类案的推理路径。并以 <|begin_expand_result|> 扩展得到的罪名，类案的推理路径 <|end_expand_result|>的格式返回给你。\n"
        "只能使用一次扩展工具，且在得到初步的预测候选罪名后立即调用。\n\n"
        "- 搜索工具，当遇到多个匹配的罪名，进行网络搜索得到对罪名的区分理解，写入 <|begin_search_query|> 如何区分[罪名A]与[罪名B] <|end_search_query|>\n"
        "然后，系统将搜索并分析相关的网页，并以 <|begin_search_result|> ...搜索结果... <|end_search_result|> 的格式将有用信息返回给你。\n"
        f"可进行多次搜索，搜索尝试的最大次数限制为 {MAX_SEARCH_LIMIT}。\n\n"
        "示例：\n"
        "Assistant思考步骤：\n"
        "- 我需要对过失致人死亡罪、过失以危险方法危害公共安全罪、重大责任事故罪进行扩展。\n"
        "Assistant：\n"
        "<|begin_expand|>过失致人死亡罪, 过失以危险方法危害公共安全罪, 重大责任事故罪<|end_expand|>\n"
        "(系统返回扩展结果)<|begin_expand_result|> 扩展得到的罪名，类案的推理路径 <|end_expand_result|>\n\n"
        "Assistant思考步骤：\n"
        "- 我需要找出过失致人死亡罪和重大责任事故罪的主要区别。\n"
        "Assistant：\n"
        "<|begin_search_query|>区分过失致人死亡罪和重大责任事故罪<|end_search_query|>\n"
        "(系统返回搜索结果)<|begin_search_result|> ...搜索结果... <|end_search_result|>\n\n"
        "注意：\n"
        "- 使用 <|begin_expand|> 请求扩展罪名，并以 <|end_expand|> 结束。\n"
        "- 当完成罪名扩展后，继续你的推理。\n\n"
        "- 使用 <|begin_search_query|> 请求网络搜索，并以 <|end_search_query|> 结束。\n"
        "- 当完成信息检索后，继续你的推理。\n\n" 
    )


def get_search_agent_user_prompt(fact, criminal, model_name):
    if model_name == 'qwq':
        user_prompt = (
            "你要做的任务是相关法条预测和罪名预测。\n"
            f"本案事实：\n{fact}\n"
            f"利用你的法律知识和外部知识，为{criminal}中每个罪犯预测相关法条和罪名。\n"
            "输出格式为:【罪犯1:法条:……,罪名:……】【罪犯2:……】……"
            "你可以按照以下步骤进行：\n"
            "1. 仔细分析给定案件的案情描述。\n"
            "2. 得到所有可能的预测结果。\n"
            "3. 利用已有的工具对所有的候选罪名进行扩展，从而得到更全面的候选罪名以及类案的推理路径。\n"
            "4. 参考类案的推理路径，用三段论的方式分析每一个候选罪名是否适用于当前案件。\n"
            "5. 当遇到多个候选罪名，进行网络搜索得到对罪名的区分理解，从而选出最适合本案的罪名。如：如何区分过失致人死亡罪和重大责任事故罪？\n"
            "6. 对搜索结果和案件事实按照三段论的方式继续推理，若搜索结果中存在不明确的知识可多次进行网络搜索，如：重大责任事故罪中的生产作业是什么？最终给出一个最合适的法条和罪名预测结果。\n"
            "注意：一定要基于当前案件的事实去推理。\n"
        )
    else:
        user_prompt = (
            "你要做的任务是对法律案件进行相关法条预测和罪名预测。\n"
            "给定一段案情描述，请利用你的法律知识和外部知识一步一步地思考，预测本案的相关法条和罪名，只需得到一个最相关的法条和罪名。\n"
            "你可以按照以下步骤进行：\n"
            "1. 仔细分析给定案件的案情描述。\n"
            "2. 得到所有可能的预测结果。\n"
            "3. 利用已有的工具对所有的候选罪名进行扩展，从而得到更全面的候选罪名以及类案的推理路径。\n"
            "4. 参考类案的推理路径，用三段论的方式分析每一个候选罪名是否适用于当前案件。\n"
            "5. 当遇到多个候选罪名，进行网络搜索得到对罪名的区分理解，从而选出最适合本案的罪名。如：如何区分过失致人死亡罪和重大责任事故罪？\n"
            "6. 对搜索结果和案件事实按照三段论的方式继续推理，若搜索结果中存在不明确的知识可多次进行网络搜索，如：重大责任事故罪中的生产作业是什么？最终给出一个最合适的法条和罪名预测结果。\n"
            "注意：一定要基于当前案件的事实去推理。\n"
            #"搜索的问题围绕法条和罪名，不要根据案件事实去搜索。\n"
            # 整理搜索内容，按照三段论的方式结合搜索结果和案件事实继续推理（把搜索结果作为大前提，案件事实作为小前提）
            #"把你的最终答案放在方括号【】中，格式：本案的相关法条是：第*条，本案的罪名是：*罪。\n"
            f"案情描述：{fact}\n"
            f"被告人：{criminal}\n"
        )
    return user_prompt

def get_search_o1_user_prompt(fact, criminal, model_name):
    if model_name == 'qwq':
        user_prompt = (
            "你要做的任务是对法律案件进行相关法条预测和罪名预测。\n"
            "给定一段案情描述，请利用你的法律知识和外部知识预测本案的相关法条和罪名，只需得到一个最相关的法条和罪名。当遇到多个候选罪名，进行网络搜索得到对罪名的区分理解，从而选出最适合本案的罪名。\n"
            f"案情描述：{fact}\n"
            f"被告人：{criminal}\n"
        )
    else:
        user_prompt = (
            "你要做的任务是对法律案件进行相关法条预测和罪名预测。\n"
            "给定一段案情描述，请利用你的法律知识和外部知识一步一步地思考，预测本案的相关法条和罪名，只需得到一个最相关的法条和罪名。当遇到多个候选罪名，进行网络搜索得到对罪名的区分理解，从而选出最适合本案的罪名。\n"
            f"案情描述：{fact}\n"
            f"被告人：{criminal}\n"
        )
    return user_prompt


def web_search_summarize_instruction(search_query, document):
    return f"""**任务说明**
        你现在的任务是：基于当前的搜索问题和检索到的网页内容，分析网页信息并提取总结有用的内容。

        **具体步骤：**

        1. **阅读并分析网页内容**
        - 仔细阅读检索到的网页内容
        - 识别与 **当前搜索问题** 相关且有助于对原始问题进行推理的信息

        2. **提取关键信息**
        - 从搜索到的网页内容中提取有助于推进 **前面推理步骤** 的信息
        - 确保信息准确、相关

        3. **总结关键信息**
        - 对提取的关键信息进行总结
        - 对于混淆罪名的区分理解，从犯罪四要件的角度去总结。

        例如：敲诈勒索罪与强迫交易罪的区别是什么?

        (1)客体不同。强迫交易罪侵犯的客体是复杂客体，即市场交易秩序和他人的人身权、财产权或其它合法权益;而敲诈勒索罪侵犯的客体是简单客体，即公私财产的所有权。

        (2)客观方面不同。强迫交易罪行为人对被害人可使用暴力、威胁方法，而敲诈勒索罪则只能使用威胁、要挟方法，若行为人当面对被害人使用暴力，则超出了敲诈勒索罪的范围，此其一。其二，本罪行为人在强迫对方达成交易后一般会给付对方一定数额的货币或商品作为代价，而敲诈勒索罪的行为人则完全是无偿占有被害人财物。

        (3)主观方面不同。强迫交易罪行为人实施强迫交易行为主观上是为了达成交易，牟取不法利益，而敲诈勒索罪行为人主观上则是为了非法占有公私财物。

        (4)主体不同。强迫交易罪自然人和单位均可构成，而敲诈勒索罪只能由自然人构成。

        4. **输出格式要求**
        - 输出以`**最终获取的信息**`开头的信息，如下所示：
        **最终获取的信息**

        [总结的信息]


        **输入：**
        - **当前搜索问题**：
        {search_query}

        - **搜索到的网页内容**：
        {document}

        现在你应该开始分析网页内容，并根据当前的搜索问题“{search_query}”提取总结有用的信息。请遵循上述步骤进行推理，并确保输出格式正确。
        """ 

def summarize_prompt(predict):
    #return f"用<>括起来的是本案事实，用---括起来的是模型利用自己的知识对案例进行一步一步地分析得到的相关法条和罪名预测结果，请根据本案事实和预测结果，从中总结提取出最相关的一个法条和罪名,重点关注用()括起来的罪犯的犯罪行为。输出格式为：本案的相关法条是：第*条，本案的罪名是：*罪。请严格按照该输出格式生成结果。\n"+f"<"+fact+f">"+f"\n---"+predict +f"---\n"+f"("+criminals +f")\n"
    return (
        "你是一个擅长提取结论并进行格式化输出的专家。"
        "给定一段对案件相关法条和罪名的预测内容，理解该内容，从中总结提取出一个最相关的法条和罪名。"
        "把你的最终答案放在方括号【】中，格式：本案的相关法条是：第*条，本案的罪名是：*罪。"
        f"预测内容：{predict}"
    )