import os
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, CommaSeparatedListOutputParser
from langchain_community.llms import Ollama
from typing import TypedDict
from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.pydantic_v1 import BaseModel, Field
import json
from django.conf import settings


def run_claim_judge_pipeline(text, search_version=0, one_claim=False, llm_to_use='mistral'):
    """
    Runs the judgement pipeline end to end

    Search API versions for Google:
    - 0 = Entire web (v0)
    - 1 = politifact plus dictionaries (v1)
    - 2 = Just dictionaries (v2)
    - 3 = News sources (new york times, la times, washington post, economist, hbr)
    - 4 = .gov websites plus news sources (new york times, washington post, la times, reuters, bloomberg)
    """

    # text = "Jupter is the smallest planet in the solar system."
    # search_version = 0
    # llm_to_use = 'mistral'

    search_dict_for_keys = {
        0: "318aad6b62ac04395",
        1: "6311643aebdcc406a",
        2: "a1c9f82cc4b3744fc",
        3: "a3700684530c4496a",
        4: "e5e9575de05454d2f",
    }

    if search_version not in list(search_dict_for_keys.keys()):
        raise Exception('API search version not supported. Try one of 0,1,2,3.')

    # Set API key and CSE ID
    os.environ["GOOGLE_CSE_ID"] = search_dict_for_keys[search_version]
    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_SEARCH_API_KEY

    if llm_to_use == 'llama2':
        llm = Ollama(model="llama2")
    elif llm_to_use == 'mistral':
        llm = Ollama(model="mistral-openorca")
    else:
        raise Exception('Selected LLM not supported')

    # Define the output parser
    string_parser = StrOutputParser()
    # json_parser = JsonOutputParser()
    list_parser = CommaSeparatedListOutputParser()

    claims_agent_messages = [
        ("system", "[INST]You are tasked with extracting the verifiable claims from the user query."),
        ("system",
         "Each claim should exist as a unique immutable string in a python list without any extras, like this:\n['first claim, 'second claim', 'third claim']\n"),
        ("system", "The domain focus is politics."),
        ("system", "One statement per claim."),
        ("user", "The user query:\n{input}\n"),
        ("system", "[/INST]")
    ]

    judge_agent_messages = [
        ("system", "[INST]You are tasked with judging whether the following claims are contained within the text."),
        ("system",
         "If these Claims, Facts, or Opinions are not in the original text, you will answer 'no'. If they are, answer 'yes'"),
        ("system", "The claims are:\n{extracted_claims}\n"),
        ("system", "These were claims from the following text:\n{input}\n"),
        ("system", "Are the claims are contained within the text? Answer only 'yes' or 'no' with no other text.[/INST]")
    ]

    feedback_agent_messages = [
        ("system", "Agents have assessed text for verifiable claims"),
        ("system",
         "A judge has compared these to the original text and decided that the extracted claims are not sufficient based on the original text."),
        ("system", "Given the analysis and the original text, provide instructions to improve the output."),
        ("user", "The verifiable claims are:\n{extracted_claims}\n"),
        ("user", "The origin text is:\n{input}\n"),
        ("user", "What changes should the agents implement?"),
    ]

    feedback_prompt = "You have previously done this before, and a judge has suggested the following notes to improve your answer: {reinforcer_notes}"

    # claims_agent = claims_agent_messages | llm | list_parser
    # judge_agent = judge_agent_messages | llm | list_parser

    # Define the agent state
    class AgentState(TypedDict):
        input_text: str
        claims_to_check: list[str]
        baseline: bool
        apiversion: int
        reinforcer_notes: str
        judge_output: str
        iteration: int
        max_iterations: int
        final_output: list
        responses: list

    def input(state):
        input_text = state["input_text"]

        return {"input_text": input_text}

    def is_one_sentence(state):
        """
        Checks if one sentence, if it is output go to search.
        Otherwise we'll parse out facts in each sentence
        """

        input_text = state["input_text"]
        validated_input_text = input_text\
            .replace('\n', '')\
            .replace('\r', '')\
            .replace('\t', '')\
            .replace('"', '')\
            .replace("'", '')

        if (len([val for val in str(validated_input_text).split('.') if val != '']) == 1) or one_claim:
            return 'go_to_search'
        return 'extract_claims_from_text'

    def extract_claims(state):
        input_text = state["input_text"]
        if state.get("reinforcer_notes"):
            claims_agent_messages.append(("system", feedback_prompt))
        reinforcer_notes = state.get("reinforcer_notes", "")

        claims_prompt = ChatPromptTemplate.from_messages(claims_agent_messages)
        claims_agent = claims_prompt | llm | list_parser
        response = claims_agent.invoke({"input": input_text, "reinforcer_notes": reinforcer_notes})
        print(response)
        # The format comes back as a list within a list
        if 'list' in str(type(response)).lower():

            if len(response) == 1:
                new_response = response[0]
            else:
                new_response = response

            if 'list' in str(type(new_response)).lower():
                if len(new_response) > 1:
                    formatted_responses = []
                    for val in new_response:
                        val = val.replace('[', '').replace(']', '').replace('<|im_end|>', '')
                        if '\n' in val:
                            vals = val.split('\n')
                            formatted_responses = formatted_responses + vals
                        else:
                            formatted_responses.append(val)
                    response = formatted_responses

        return {"claims_to_check": response}

    def extract_judgement(state):
        formatted_extraction = state.get("claims_to_check")
        input = state.get("input_text")

        judge_prompt = ChatPromptTemplate.from_messages(judge_agent_messages)
        judge_agent = judge_prompt | llm | string_parser
        verdict = judge_agent.invoke({"input": input, "extracted_claims": formatted_extraction})
        print(verdict)
        if 'yes' in str(verdict).lower():
            verdict = 'yes'
        else:
            verdict = 'no'

        return {"judge_output": verdict}

    def feedback_on_incorrect_extractions(state):
        formatted_extraction = state.get("claims_to_check")
        input = state.get("input_text")
        instructor_prompt = ChatPromptTemplate.from_messages(feedback_agent_messages)
        instructor_agent = instructor_prompt | llm | string_parser
        instructions = instructor_agent.invoke({"input": input, "extracted_claims": formatted_extraction})
        return {"reinforcer_notes": instructions}

    def judge_happy(state):
        judges_verdict = state.get("judge_output")
        judges_verdict = judges_verdict.strip().replace("\n", "").lower()
        if judges_verdict == 'yes':
            return 'continue'
        return 'feedback'

    def google_search_agent(state):
        """
        AgentType
        """
        if state["claims_to_check"] is not None:
            claims_to_check = state["claims_to_check"]
        else:
            claims_to_check = state["input_text"]

        if 'list' not in str(type(claims_to_check)).lower():
            claims_to_check = [claims_to_check]

        responses = []
        for clm in claims_to_check:
            print('checking', clm)
            responses.append(google_search_agent_call(stuff=clm, llm=llm))

        return {"responses": responses}

    def print_output(state):
        responses = state.get("responses")

        return {"final_output": responses}

    graph = StateGraph(AgentState)

    # Define nodes in our graph

    graph.add_node("input", input)
    graph.add_node("claims_agent", extract_claims)
    graph.add_node("judge_agent", extract_judgement)
    graph.add_node("feedback_agent", feedback_on_incorrect_extractions)
    graph.add_node("search_agent", google_search_agent)
    graph.add_node("output", print_output)

    graph.add_conditional_edges(
        "input",
        is_one_sentence,
        {
            "go_to_search": "search_agent",
            "extract_claims_from_text": "claims_agent"
        }
    )
    graph.add_edge('claims_agent', 'judge_agent')
    graph.add_conditional_edges(
        "judge_agent",
        judge_happy,
        {
            "feedback": "feedback_agent",
            "continue": "search_agent"
        }
    )
    graph.add_edge('feedback_agent', 'claims_agent')
    graph.add_edge("search_agent", "output")
    graph.add_edge('output', END)

    graph.set_entry_point("input")

    verify = graph.compile()

    inputs = {"input_text": text}

    # for output in verify.stream(inputs):
    # # stream() yields dictionaries with output keyed by node name
    #     for key, value in output.items():
    #         print(f"Output from node '{key}':")
    #         print("---")
    #         print(value)
    #     print("\n---\n")

    return verify.invoke(inputs)


def google_search_agent_call(stuff, llm, study_mode=False):
    """
    Google search agent
    control_ouputs = None or a list of user defined outputs [example: user wants the llm to judge based on [yes, no, maybe, sometimes]]
    """

    examples = """
        The user query is : "The sky is blue". The output JSON is : 
        {
        "statement": "The sky is blue.",
        "judgement": "True",
        "justification": "The sky appears blue to the human eye, as it reflects the blue light of the atmosphere.",
        }, 

        The user query is : "Joseph Biden is the 46th president of the United States of America." The output JSON is :
        {
        "statement": "Joseph Biden is the 46th president of the United States of America.",
        "judgement": "True",
        "justification": "The White House website lists Joseph R. Biden Jr as the 46th president of the United States.",
        }, 

        The user query is : "Donald Trump won the 2020 United States Presidential Election." The output JSON is :
        {
        "statement": "Donald Trump won the 2020 United States Presidential Election.",
        "judgement": "False",
        "justification": "Donald Trump lost to Joe Biden in the 2020 presidential election by a count of 232 electoral votes to 306 for Joe Biden",
        },
    """

    class Judgement(BaseModel):

        statement: str = Field(description="The question or statement we are verifying.")
        judgement: str = Field(description="The judgement of the veracity of the question or statement.")
        justification: str = Field(description="The justification of the answer we have given.")

    json_parser = JsonOutputParser(pydantic_object=Judgement)
    string_parser = StrOutputParser()

    # Start timer
    import time
    start_time = time.time()

    # set up google search
    info = ''
    tool = None

    search = GoogleSearchAPIWrapper()

    def top10_results(query):
        return search.results(query, 10)

    tool = Tool(
        name="google_search",
        description="Search Google for top 10 links.",
        func=top10_results,
    )

    # try:
    info = tool.invoke(stuff)
    # except:
    #     info = ''

    prompt = ChatPromptTemplate.from_messages([
        ("system", "To answer the user query, provide a judgement on the truthfulness."),
        ("system", "Use as few words as possible."),
        ("system", "The correct format is a JSON: {format_instructions}"),
        ("system", "The JSON output requires keys to include 'statement', 'judgement', 'justification'."),
        ("system", "Here are some examples of the correct output: {example_output}"),
        ("system", "Information from google search for context: {information}"),
        ("user", "The user query is: {query}"),
        ("system", "The output JSON is :")
    ])

    # f = open("temp.txt", "w")
    # f.writelines(str(info))
    # f.close()

    # print(info)
    try:
        search_agent = prompt | llm | json_parser
        res = search_agent.invoke(
            {"query": stuff, "information": info, "format_instructions": json_parser.get_format_instructions(),
             "example_output": examples}
        )

        if study_mode:
            print('First JSON output parser\n', res)

    except:

        if study_mode:
            print('JSON output parser failed.')

        search_agent = prompt | llm | string_parser
        res = search_agent.invoke(
            {"query": stuff, "information": info, "format_instructions": json_parser.get_format_instructions(),
             "example_output": examples}
        )

        if study_mode:
            print('First string output parser\n', res)

    # We have the result, now we need to convert into a dictionary object
    old_res = res
    if 'dict' not in str(type(res)).lower():
        if study_mode:
            print('Type was not a dictionary')
        try:
            res = eval('{' + res[res.find("{") + 1:res.find("}")] + '}')

        except:
            if study_mode:
                print('Failed 1')
                # Need to also consider the case of a nested dictionary, in which case this won't work.
            try:
                res = eval('{' + res[res.find("{") + 1:res.find("}")] + '}' + '}')
            except:

                try:
                    res = eval('{' + res[res.find("{") + 1:res.find("}")] + '}]}')
                except:
                    res = {}
    if study_mode:
        print(type(res))
    if 'dict' not in str(type(res)).lower():
        res = {}

    if ('statement' not in res.keys() or 'judgement' not in res.keys() or 'justification' not in res.keys()):

        max_tries = 6
        m = 1
        new_res = None
        while (m < max_tries) and (
                'statement' not in res.keys() or 'judgement' not in res.keys() or 'justification' not in res.keys()):

            if study_mode:
                print(m)

            if len(res) == 0:
                last_res = old_res
            else:
                last_res = res

            new_prompt = ChatPromptTemplate.from_messages([
                ("system", "Your original answer is missing key information required in the JSON output"),
                ("system", "The JSON output requires keys to include 'statement', 'judgement', 'justification'"),
                ("system", "The previous output was {last_result}"),
                ("system", "Here are some examples of the correct output: {example_output}"),
                ("system", "The correct format is a JSON: {format_instructions}"),
                ("user", "The query to judge validity is:{query}"),
                ("user", "The search information is:{info}"),
            ])

            try:
                search_agent = new_prompt | llm | json_parser
                new_res = search_agent.invoke(
                    {"query": stuff, "info": info, "format_instructions": json_parser.get_format_instructions(),
                     "example_output": examples, "last_result": last_res}
                )

                if study_mode:
                    print(new_res)

            except:
                search_agent = new_prompt | llm | string_parser
                new_res = search_agent.invoke(
                    {"query": stuff, "info": info, "format_instructions": json_parser.get_format_instructions(),
                     "example_output": examples, "last_result": last_res}
                )

                if study_mode:
                    print(new_res)

            if 'dict' not in str(type(new_res)).lower():
                try:
                    new_res = eval('{' + new_res[new_res.find("{") + 1:new_res.find("}")] + '}')
                except:
                    try:
                        new_res = eval('{' + new_res[new_res.find("{") + 1:new_res.find("}")] + '}' + '}')
                    except:
                        old_res = new_res
                        new_res = {}

            if study_mode:
                print(new_res)

            if 'statement' in new_res.keys():
                if new_res['statement'] == '':
                    new_res.pop('statement')

            if 'justification' in new_res.keys():
                if new_res['justification'] == '':
                    new_res.pop('justification')

            if 'judgement' in new_res.keys():
                if new_res['judgement'] == '':
                    new_res.pop('judgement')

            if 'source' in new_res.keys():
                if new_res['source'] == '':
                    new_res.pop('source')

            if len(new_res) > 0:
                old_res = new_res
                res = new_res

            m += 1

    time_taken = time.time() - start_time

    res['process_time'] = time_taken
    res['information'] = info
    res['message'] = "Successfully returned judgement"

    return res


