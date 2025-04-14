#Importing the libraries
import os
import time
from bs4 import BeautifulSoup
import re
import requests
import base64
import json
import yfinance as yf
import langchain
from langchain.agents import Tool, initialize_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.callbacks import StreamlitCallbackHandler
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('bcg_light.png')
st.header('Stock Recommendation System')
#importing api key as environment variable

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

st.sidebar.write('This tool provides recommendation based on the RAG & ReAct Based Schemes:')
lst = ['Get Ticker Value',  'Fetch Historic Data on Stock','Get Financial Statements','Scrape the Web for Stock News','LLM ReAct based Verbal Analysis','Output Recommendation: Buy, Sell, or Hold with Justification']

s = ''

for i in lst:
    s += "- " + i + "\n"

st.sidebar.markdown(s)

if openai_api_key:
    llm=ChatOpenAI(temperature=0,model_name='gpt-4-turbo',openai_api_key=openai_api_key)

    #Get Historical Stock Closing Price for Last 1 Year
    def get_stock_price(ticker):
        if "." in ticker:
            ticker = ticker.split(".")[0]
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        df = df[["Close","Volume"]]
        df.index=[str(x).split()[0] for x in list(df.index)]
        df.index.rename("Date",inplace=True)
        return df.to_string()


    #Get News From Web Scraping
    def google_query(search_term):
        if "news" not in search_term:
            search_term = search_term+" stock news"
        url = f"https://www.google.com/search?q={search_term}"
        url = re.sub(r"\s","+",url)
        return url

    #Get Recent Stock News
    def get_recent_stock_news(company_name):
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
        g_query = google_query(company_name)
        res=requests.get(g_query,headers=headers).text
        soup = BeautifulSoup(res,"html.parser")
        news=[]
        for n in soup.find_all("div","n0jPhd ynAwRc tNxQIb nDgy9d"):

            news.append(n.text)
        for n in soup.find_all("div","IJl0Z"):
            news.append(n.text)

        if len(news) > 6:
            news = news[:4]
        else:
            news = news
        
        news_string=""
        for i,n in enumerate(news):
            news_string+=f"{i}. {n}\n"
        top5_news="Recent News:\n\n"+news_string
        
        return top5_news

    #Get Financial Statements
    def get_financial_statements(ticker):
        if "." in ticker:
            ticker = ticker.split(".")[0]
        else:
            ticker=ticker
        company = yf.Ticker(ticker)
        balance_sheet = company.balance_sheet
        if balance_sheet.shape[1]>3:
            balance_sheet = balance_sheet.iloc[:,:3]
        balance_sheet = balance_sheet.dropna(how="any")
        balance_sheet = balance_sheet.to_string()
        return balance_sheet

    #Initialize DuckDuckGo Search Engine
    search=DuckDuckGoSearchRun()     
    tools = [
    Tool(
        name="Stock Ticker Search",
        func=search.run,
        description="Use only when you need to get stock ticker from internet, you can also get recent stock related news. Dont use it for any other analysis or task"

    ),
    Tool(
        name = "Get Stock Historical Price",
        func = get_stock_price,
        description="Use when you are asked to evaluate or analyze a stock. This will output historic share price data. You should input the stock ticker to it"

    ),
    Tool(
        name="Get Recent News",
        func= get_recent_stock_news,
        description="Use this to fetch recent news about stocks"
    ),
    Tool(
        name="Get Financial Statements",
        func=get_financial_statements,
        description="Use this to get financial statement of the company. With the help of this data company's historic performance can be evaluated. You should input stock ticker to it"
    )
    ]

    zero_shot_agent=initialize_agent(
        llm=llm,
        agent="zero-shot-react-description",
        tools=tools,
        verbose=True,
        max_iteration=4,
        return_intermediate_steps=False,
        handle_parsing_errors=True
    )

    #Adding predefine evaluation steps in the agent Prompt
    stock_prompt="""You are a financial advisor. Give stock recommendations for given query.
    Everytime first you should identify the company name and get the stock ticker symbol for the stock.
    Answer the following questions as best you can. You have access to the following tools:

    Get Stock Historical Price: Use when you are asked to evaluate or analyze a stock. This will output historic share price data. You should input the stock ticker to it 
    Stock Ticker Search: Use only when you need to get stock ticker from internet, you can also get recent stock related news. Dont use it for any other analysis or task
    Get Recent News: Use this to fetch recent news about stocks
    Get Financial Statements: Use this to get financial statement of the company. With the help of this data company's historic performance can be evaluaated. You should input stock ticker to it

    steps- 
    Note- if you fail in satisfying any of the step below, Just move to next one
    1) Get the company name and search for the "company name + stock ticker" on internet. Dont hallucinate extract stock ticker as it is from the text. Output- stock ticker. If stock ticker is not found, stop the process and output this text: This stock does not exist
    2) Use "Get Stock Historical Price" tool to gather stock info. Output- Stock data
    3) Get company's historic financial data using "Get Financial Statements". Output- Financial statement
    4) Use this "Get Recent News" tool to search for latest stock related news. Output- Stock news
    5) Analyze the stock based on gathered data and give detailed analysis for investment choice. provide numbers and reasons to justify your answer. Output- Give a single answer if the user should buy,hold or sell. You should Start the answer with Either Buy, Hold, or Sell in Bold after that Justify.

    Use the following format:

    Question: the input question you must answer
    1. If you need to perform an action to answer the user's question, use the following format:
    '''
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [Get Stock Historical Price, Stock Ticker Search, Get Financial Statements, Get Recent News]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    '''

    2. If your thought contains Buy, Hold, or Sell, then you stop looking for more tools, then you MUST use the following format:
    '''
    Final Thought: Do I need to use a tool? No
    Final Answer: the final answer to the original input question
    '''
    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""

    zero_shot_agent.agent.llm_chain.prompt.template=stock_prompt

    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = zero_shot_agent(f'Is {prompt} a good investment choice right now?', callbacks=[st_callback])
            st.write(response["output"])


