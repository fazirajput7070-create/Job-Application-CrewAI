from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, LLM
from crewai_tools import TavilySearchTool
search_tool = TavilySearchTool()



llm = LLM(
    model="ollama/llama3.2"
)


job_researcher = Agent(
    role="Job researcher",
    goal="Analyze a job description and identify the key requirements, skills, and qualifications needed for the position.",
    backstory="You are an experienced job researcher who carefully examine job descriptions and identifies the most important requirements.",
    tools=[search_tool],
    llm=llm
)

personal_profiler = Agent(
    role="Personal Profiler for Engineers",
    goal="Analyze the candidate's skills, experienced, projects, and background to create a clear professional profile.",
    backstory="You are an experienced personal profiler who creates detailed professional profiles for engineers.",
    llm=llm
)

resume_strategist = Agent(
    role="Resume Strategist",
    goal="Create a strategy for tailoring the candidate's resume to match the requirements of the target job.",
    backstory="You are an experienced resume strategist who knows how to highlight a candidate's strongest and most relevant skills for a specific job.",
    llm=llm
)

interview_preparer = Agent(
    role="Interview Preparer",
    goal="Prepare the candidate for the job interview by creating releavent interview questions and preparation guidance.",
    backstory="You are an experienced interciew coach who helps candidates prepare for job interciews by providing them with relevant interview questions based on job requirements and their backgrounds.",
    llm=llm
)


job_researcher_task =Task(
    description="Research current job opportunities for a software engineer based on the candidate's skills and experienced. Identify companies that are hiring and collect the relevant job requirements ",
    expected_output="A clear and organized list of job opprtunities for a software engineer, including the company name, job titiles, required skills, experienced requirements, and other  important details.",
    agent=job_researcher
)


personal_profiler_task = Task(
    description="Analyze the candidate's skills, experienced, projects, and background to create a clear professional profile.",
    expected_output="A detailed professional profile containing the candidate's skills,experienced, projects, strengths, and relevant qualifications.",
    agent=personal_profiler
)


resume_strategist_task = Task(
    description="Create a strategy to tailor the candidate's resume according  to the target job requirements. Identify which skills, experience, and projects should be highlighted.",
    expected_output="A clear resume strategy explaining how the candidate's resume be customized for the target job.",
    context=[job_researcher_task, personal_profiler_task],
    agent=resume_strategist
)

interview_preparer_task = Task(
    description="Prepare the candidate for the target job interview by creating relevant interview questions adn practical preparation guidance.",
    expected_output="A personalized interview preparation guide containing interview questions, important topics to prepare, and useful guidance for the candidate.",
    context=[job_researcher_task, personal_profiler_task,resume_strategist_task],
    agent=interview_preparer
)


crew=Crew(
    agents=[job_researcher, personal_profiler, resume_strategist, interview_preparer],
    tasks=[job_researcher_task, personal_profiler_task, resume_strategist_task, interview_preparer_task]
)

result = crew.kickoff()

print(result)

