import pandas as pd
import os
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from typing import TypedDict
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from django.conf import settings


def run_claim_judge_pipeline(text, one_claim=False, search_version=0, llm_to_use='llama2', outputs=None,
                             baseline=False):
    """
    Runs the judgement pipeline end to end

    Search API versions for Google:
    - 0 = Entire web (v0)
    - 1 = politifact plus dictionaries (v1)
    - 2 = Just dictionaries (v2)
    - 3 = News sources (new york times, la times, washington post, economist, hbr)
    """

    search_dict_for_keys = {
        0: "318aad6b62ac04395",
        1: "6311643aebdcc406a",
        2: "a1c9f82cc4b3744fc",
        3: "a3700684530c4496a",
    }

    if search_version not in list(search_dict_for_keys.keys()):
        raise Exception('API search version not supported. Try one of 0,1,2,3.')

    # Set API key and CSE ID
    os.environ["GOOGLE_CSE_ID"] = search_dict_for_keys[search_version]
    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_SEARCH_API_KEY

    if llm_to_use == 'llama2':
        llm = Ollama(model="llama2")
    else:
        raise Exception('Selected LLM not supported')

    if one_claim or (len([val for val in str(text).split('.') if val != '']) == 1):
        # We just have one sentence that can directly be fed into the google search engine
        # print('simple')
        # verifiable = simple_judge(text, llm)
        # print('verifiable:',verifiable)
        # if 'yes' in verifiable or 'sure' in verifiable:

        # We have determined it to be verifiable
        data = pd.DataFrame(data={'Category': 'Verifiable', 'Text': str(text).replace('* ', '')}, index=[0])

        # else:
        #    data = pd.DataFrame()

    # Check whether the input is just one sentence:

    else:

        agent_response = extract_claims(text, llm)

        datas = []

        facts = False
        vclaims = False
        uclaims = False

        for i in agent_response.split('\n'):
            print(str(i))
            if str(i) != '':

                if str(i).lower().replace(':', '').replace('-', '') in ['fact claims', 'facts', 'factual claims',
                                                                        'factual statements', 'fact',
                                                                        'statements of fact']:
                    facts = True
                    vclaims = False
                    uclaims = False
                    continue

                if str(i).lower().replace(':', '').replace('-', '') in ['verifiable claims', 'verifiable', 'claims']:
                    vclaims = True
                    facts = False
                    uclaims = False
                    continue

                if str(i).lower().replace(':', '').replace('-', '') in ['opinions', 'unverifiable claims', 'unverified',
                                                                        'not verifiable']:
                    uclaims = True
                    facts = False
                    vclaims = False
                    continue

                if facts:
                    temp_data = datas.append(
                        pd.DataFrame(data={'Category': 'Fact', 'Text': str(i).replace('* ', '')}, index=[0]))

                if vclaims:
                    temp_data = datas.append(
                        pd.DataFrame(data={'Category': 'Verifiable', 'Text': str(i).replace('* ', '')}, index=[0]))

                if uclaims:
                    temp_data = datas.append(
                        pd.DataFrame(data={'Category': 'Unverifiable', 'Text': str(i).replace('* ', '')}, index=[0]))

        data = pd.concat(datas)

    print(data)

    if len(data) > 0:

        responses = []
        for i, row in data[data['Category'].isin(['Verifiable', 'Fact'])].iterrows():
            # print(row['Text'])
            if baseline:
                responses.append(google_search_agent(stuff=str(row['Text']), llm=llm, response_option='minimal',
                                                     model_option='baseline', return_information=False,
                                                     control_outputs=outputs))
            else:
                responses.append(
                    google_search_agent(stuff=str(row['Text']), llm=llm, response_option='minimal', model_option='rag',
                                        return_information=True, control_outputs=outputs))
            # print(responses)

        v = {k: [dic[k] for dic in responses] for k in responses[0]}
        v['message'] = 'Successfully performed judgements.'
        return v

    else:
        return {'message': 'There were no verifiable claims detected.'}


def extract_claims(text, llm):
    """
    From a large text entry, supports the extraction of claims from the text
    """

    # llm = Ollama(model="llama2")

    # Define the output parser
    output_parser = StrOutputParser()

    fact_agent_messages = [
        ("system",
         "You are tasked with extracting from text the claims of specific values, figures, or facts. Exclude any opinions or personal preferences."),
        ("system", "You answer in short sentences."),
        ("system", "The domain focus is politics."),
        ("system", "Each reply is a bullet point."),
        ("user", "The text is: {input}")
    ]

    claims_agent_messages = [
        ("system",
         "You are tasked with extracting from text the verifiable claims. If it is possible to verify, include it. Exclude any opinions or personal preferences."),
        ("system", "You answer in short sentences."),
        ("system", "The domain focus is politics."),
        ("system", "Each reply is a bullet point."),
        ("user", "The text is: {input}")
    ]

    opinions_agent_messages = [
        ("system", "You are tasked with extracting all opinions from the text which are not verifyable."),
        ("system", "You answer in short sentences."),
        ("system", "The domain focus is politics."),
        ("system", "Each reply is a bullet point."),
        ("user", "The text is: {input}")
    ]

    formatter_agent_messages = [
        ("system", "You are very organized and take inputs from different sources and make well formatted lists."),
        ("system",
         "You take input from three sources, fact claims, opinions, and verifiable claims, and organize them. You do not add anything other than what is given to you."),
        ("system",
         "You always output lists in the below format:\n Fact Claims:\n - fact claim 1\n  - fact claim 2\n\n Verifiable Claims:\n - verifiable claim 1\n - verifiable claim 2\n\n Opinions:\n - opinion 1\n - opinion 2"),
        ("user", "Fact claims are:\n {facts}"),
        ("user", "Verifiable claims are:\n {claims}"),
        ("user", "Opinions are: \n {opinions}"),
    ]

    judge_agent_messages = [
        ("system",
         "You are tasked with judging whether the following claims are contained within the text. You will be given a set of Fact Claims, Verifiable Claims, and Opinions from a text."),
        ("system",
         "If these Claims, Facts, or Opinions are not in the original text, you will answer 'no'. If they are, answer 'yes'"),
        ("system", "You only respond with 'yes' or 'no', with no other text."),
        ("system", "Claims, Facts, and Opinions:\n{formatted_extractions}\n"),
        ("system", "These were judgements based on the following text:\n {input}\n"),
        ("system", "Are the claims are contained within the text? Answer only 'yes' or 'no' with no other text.")
    ]

    feedback_agent_messages = [
        ("system", "Three agents have assessed a text for verifiable and unverifiable claims"),
        ("system",
         "A judge has compared these to the original text and decided that the extracted claims are not sufficient based on the original text."),
        ("system",
         "The facts may not exist in the text, the opinions may not be opinions, or some other problem. Given the analysis and the original text, provide instructions to improve the output."),
        ("user", "Extracted Facts, Claims, and Opinions:\n {formatted_extractions}"),
        ("user", "Original Text:\n {input}"),
        ("user", "What changes should the agents implement?"),
    ]

    feedback_prompt = "You have previously done this before, and a judge has suggested the following notes to improve your answer: {reinforcer_notes}"

    facts_agent = fact_agent_messages | llm | output_parser
    claims_agent = claims_agent_messages | llm | output_parser
    opinions_agent = opinions_agent_messages | llm | output_parser
    judge_agent = judge_agent_messages | llm | output_parser

    # Define the agent state
    class AgentState(TypedDict):
        facts: str
        claims: str
        opinions: str
        formatted_extraction: str
        reinforcer_notes: str
        judge_output: str
        input_text: str
        iteration: int
        max_iterations: int
        final_output: str

    def extract_facts(state):
        input_text = state["input_text"]
        if state.get("reinforcer_notes"):
            fact_agent_messages.append(("system", feedback_prompt))
        reinforcer_notes = state.get("reinforcer_notes", "")

        facts_prompt = ChatPromptTemplate.from_messages(fact_agent_messages)
        facts_agent = facts_prompt | llm | output_parser
        response = facts_agent.invoke({"input": input_text, "reinforcer_notes": reinforcer_notes})
        return {"facts": response}

    def extract_opinions(state):
        input_text = state["input_text"]
        if state.get("reinforcer_notes"):
            opinions_agent_messages.append(("system", feedback_prompt))
        reinforcer_notes = state.get("reinforcer_notes", "")

        opinions_prompt = ChatPromptTemplate.from_messages(opinions_agent_messages)
        opinions_agent = opinions_prompt | llm | output_parser
        response = opinions_agent.invoke({"input": input_text, "reinforcer_notes": reinforcer_notes})
        return {"opinions": response}

    def extract_claims(state):
        input_text = state["input_text"]
        if state.get("reinforcer_notes"):
            claims_agent_messages.append(("system", feedback_prompt))
        reinforcer_notes = state.get("reinforcer_notes", "")

        claims_prompt = ChatPromptTemplate.from_messages(claims_agent_messages)
        claims_agent = claims_prompt | llm | output_parser
        response = claims_agent.invoke({"input": input_text, "reinforcer_notes": reinforcer_notes})
        return {"claims": response}

    def format_extractions(state):
        facts = state.get("facts")
        opinions = state.get("opinions")
        claims = state.get("claims")
        formatter_prompt = ChatPromptTemplate.from_messages(formatter_agent_messages)
        formatter_agent = formatter_prompt | llm | output_parser
        formatted_text = formatter_agent.invoke({"facts": facts, "claims": claims, "opinions": opinions})
        return {'formatted_extraction': formatted_text}

    def extract_judgement(state):
        formatted_extraction = state.get("formatted_extraction")
        input = state.get("input_text")

        judge_prompt = ChatPromptTemplate.from_messages(judge_agent_messages)
        judge_agent = judge_prompt | llm | output_parser
        verdict = judge_agent.invoke({"input": input, "formatted_extractions": formatted_extraction})
        return {"judge_output": verdict}

    def feedback_on_incorrect_extractions(state):
        formatted_extraction = state.get("formatted_extraction")
        input = state.get("input_text")
        instructor_prompt = ChatPromptTemplate.from_messages(feedback_agent_messages)
        instructor_agent = instructor_prompt | llm | output_parser
        instructions = instructor_agent.invoke({"input": input, "formatted_extractions": formatted_extraction})
        return {"reinforcer_notes": instructions}

    def judge_happy(state):
        judges_verdict = state.get("judge_output")
        judges_verdict = judges_verdict.strip().replace("\n", "").lower()
        print(judges_verdict)
        if judges_verdict == 'yes':
            return 'continue'
        return 'feedback'

    def print_output(state):
        return {"final_output": "HAPPY"}

    graph = StateGraph(AgentState)

    # Define nodes in our graph
    graph.add_node("facts_agent", extract_facts)
    graph.add_node("opinions_agent", extract_opinions)
    graph.add_node("claims_agent", extract_claims)
    graph.add_node("formatting_agent", format_extractions)
    graph.add_node("judge_agent", extract_judgement)
    graph.add_node("feedback_agent", feedback_on_incorrect_extractions)
    graph.add_node("output", print_output)

    graph.add_edge('facts_agent', 'opinions_agent')
    graph.add_edge('opinions_agent', 'claims_agent')
    graph.add_edge('claims_agent', 'formatting_agent')
    graph.add_edge('formatting_agent', 'judge_agent')
    graph.add_conditional_edges(
        "judge_agent",
        judge_happy,
        {
            "feedback": "feedback_agent",
            "continue": "output"
        }
    )
    graph.add_edge('feedback_agent', 'facts_agent')
    graph.add_edge('output', END)

    graph.set_entry_point("facts_agent")

    verify = graph.compile()

    inputs = {"input_text": text}

    out = None
    for output in verify.stream(inputs):
        for key, value in output.items():
            if key == '__end__':
                out = value['formatted_extraction']
                print(out)

    return out


def google_search_agent(stuff, llm, response_option=None, model_option='rag', return_information=True,
                        control_outputs=None):
    """
    Google search agent
    """

    examples = [
        {
            "statement": "The sky is blue.",
            "judgement": "True",
            "justification": "The sky appears blue to the human eye, as it reflects the blue light of the atmosphere.",
            "source": "NASA: https://spaceplace.nasa.gov/blue-sky/en/",
        },
        {
            "statement": "Joseph Biden is the 46th president of the United States of America",
            "judgement": "True",
            "justification": "The White House website lists Joseph R. Biden Jr as the 46th president of the United States.",
            "source": "White House: https://www.whitehouse.gov/about-the-white-house/presidents/",
        },
        {
            "statement": "Donald Trump won the 2020 United States Presidential Election.",
            "judgement": "False",
            "justification": "Donald Trump lost to Joe Biden in the 2020 presidential election by a count of 232 electoral votes to 306 for Joe Biden",
            "source": "Wikipedia: https://en.wikipedia.org/wiki/2020_United_States_presidential_election",
        },
        {
            "statement": "The earth is flat.",
            "judgement": "False",
            "justification": "Scientific observations of the Earth have proven the Earth is a round object that moves around the Sun.",
            "source": "NASA: https://science.nasa.gov/earth/facts/",
        },
        {
            "statement": "The universe is 6000 years old.",
            "judgement": "False",
            "justification": "The age of the universe is estimated by NASA's WMAP project to be 13.772 billion years old with an uncertainty plus or minus 59 million years.",
            "source": "Wikipedia: https://en.wikipedia.org/wiki/Age_of_the_universe",
        },
    ]

    class Judgement(BaseModel):

        statement: str = Field(description="The question or statement we are verifying.")
        judgement: str = Field(description="The judgement of the veracity of the question or statement.")
        justification: str = Field(description="The justification of the answer we have given.")
        source: str = Field(description="A source of information that supports our justification.")

    output_parser = JsonOutputParser(pydantic_object=Judgement)
    string_parser = StrOutputParser()

    # try:
    import time
    start_time = time.time()

    info = ''
    tool = None
    if model_option != 'baseline':

        search = GoogleSearchAPIWrapper()

        def top10_results(query):
            return search.results(query, 10)

        tool = Tool(
            name="google_search",
            description="Search Google for recent results.",
            func=top10_results,
        )

        # search = GoogleSearchAPIWrapper()
        # tool = Tool(
        #     name="google_search",
        #     description="Search Google for recent results.",
        #     func=search.run,
        # )

        if control_outputs is not None and len(control_outputs) > 1:

            prompt = ChatPromptTemplate.from_messages([
                ("system", "To answer the user query, provide a judgement on the veracity. Keep verbosity low."),
                ("system",
                 "The JSON output requires keys to include 'statement', 'judgement', 'justification', and 'source'"),
                ("system", "The judgement is one of the following: {controlled_judgements}"),
                ("system", "The correct format is a JSON: {format_instructions}"),
                ("system", "Here are some examples of the correct output: {example_output}"),
                ("system", "Information from google search for context: {information}"),
                ("user", "The query to judge validity is: {query}"),
            ])

        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "To answer the user query, provide a judgement on the veracity. Keep verbosity low."),
                ("system",
                 "The JSON output requires keys to include 'statement', 'judgement', 'justification', and 'source'"),
                ("system", "The correct format is a JSON: {format_instructions}"),
                ("system", "Here are some examples of the correct output: {example_output}"),
                ("system", "Information from google search for context: {information}"),
                ("user", "The query to judge validity is: {query}"),
            ])

        info = tool.invoke(stuff)

        f = open("temp.txt", "w")
        f.writelines(str(info))
        f.close()

        # print(info)
        # search_agent = prompt | llm | output_parser
        search_agent = prompt | llm | string_parser

        if control_outputs is not None:
            res = search_agent.invoke(
                {"query": stuff, "information": info, "format_instructions": output_parser.get_format_instructions(),
                 "controlled_judgements": control_outputs, "example_output": str(examples)})
        else:
            res = search_agent.invoke(
                {"query": stuff, "information": info, "format_instructions": output_parser.get_format_instructions(),
                 "example_output": str(examples)})

        print(res)

    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "To answer the user query, provide a judgement on the veracity. Keep verbosity low."),
            ("system",
             "The JSON output requires keys to include 'statement', 'judgement', 'justification', and 'source'"),
            ("system", "The correct format is a JSON: {format_instructions}"),
            ("system", "Here are some examples of the correct output: {example_output}"),
            ("user", "The query to judge validity is: {query}"),
        ])

        search_agent = prompt | llm | string_parser
        res = search_agent.invoke({"query": stuff, "format_instructions": output_parser.get_format_instructions(),
                                   "example_output": str(examples)})

    # We have the result, now we need to convert into a dictionary object
    try:
        res = eval('{' + res[res.find("{") + 1:res.find("}")] + '}')
    except Exception as e:
        res = {}

    print(res)
    if ('statement' not in res.keys()
            or 'judgement' not in res.keys()
            or 'justification' not in res.keys()
            or 'source' not in res.keys()):

        max_tries = 100
        m = 1
        new_res = None
        while (m < max_tries) and (
                'statement' not in res.keys() or 'judgement' not in res.keys() or 'justification' not in res.keys() or 'source' not in res.keys()):

            print(m)
            new_output_parser = StrOutputParser()

            # try:
            new_prompt = ChatPromptTemplate.from_messages([
                ("system", "Your original answer is missing key information required in the JSON output"),
                ("system",
                 "The JSON output requires keys to include 'statement', 'judgement', 'justification', and 'source'"),
                ("system", "The previous output was {last_result}"),
                ("system",
                 "Try to answer again and provide the right output format. No additional text outside of the JSON"),
                ("system", "The correct format is a JSON: {format_instructions}"),
                ("user", "The query to judge validity is:{query}"),
                ("user", "The search information is:{info}"),
            ])
            check_agent = new_prompt | llm | new_output_parser
            new_res = check_agent.invoke({"query": stuff, "info": info, "last_result": res,
                                          "format_instructions": output_parser.get_format_instructions()})
            print(new_res)
            try:
                new_res = eval('{' + new_res[new_res.find("{") + 1:new_res.find("}")] + '}')
            except:
                new_res = {}
            print(new_res)
            if len(new_res) > 0:
                res = new_res

            m += 1

    time_taken = time.time() - start_time

    res['process_time'] = time_taken

    if return_information:
        res['information'] = str(info)
        res['tool'] = tool
    else:
        res['information'] = ''
        res['tool'] = None

    return res


def simple_judge(text, llm):
    output_parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "To answer the user query, provide a judgement on the veracity. Keep verbosity low."),
        ("system", "The JSON output requires keys to include 'statement', 'judgement', 'justification', and 'source'"),
        ("system", "The correct format is a JSON: {format_instructions}"),
        ("user", "The query to judge validity is:{query}"),
    ])

    # Define the output parser
    output_parser = StrOutputParser()

    # Define the chain
    chain = prompt | llm | output_parser

    return str(chain.invoke({"input": text}))
