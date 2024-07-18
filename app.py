from fastapi import FastAPI, HTTPException, Response, Header
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
import os
from io import BytesIO
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  
)

GOOGLE_API_KEY = os.getenv("G-API")
CUSTOM_SEARCH_ENGINE_ID = os.getenv("SEARCH-KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PersonRequest(BaseModel):
    name: str

def search_person(name: str):
    url = f"https://customsearch.googleapis.com/customsearch/v1?q={name}&key={GOOGLE_API_KEY}&cx={CUSTOM_SEARCH_ENGINE_ID}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch search results")
    
    search_results = response.json().get('items', [])
    top_websites = ["wikipedia.org", "linkedin.com"]  # Add more websites as needed
    filtered_results = [result for result in search_results if any(site in result.get('link', '') for site in top_websites)]
    
    return filtered_results

def summarize_content(content: str):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={"type": "text"},
        temperature=0.7,
        max_tokens=250,
        messages=[
            {"role": "system", "content": "The content received will be based on a web search. Summarize it and provide the information which is relevant and makes sense."},
            {"role": "user", "content": content}
        ]
    )
    summary = response.choices[0].message.content
    return summary

def search_image(name: str):
    url = f"https://customsearch.googleapis.com/customsearch/v1?q={name}&key={GOOGLE_API_KEY}&cx={CUSTOM_SEARCH_ENGINE_ID}&searchType=image&num=10"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch image results")
    
    image_results = response.json().get('items', [])
    
    if not image_results:
        return None
    
    for image_result in image_results:
        image_url = image_result.get('link')
        if image_url and (image_url.endswith(".jpg") or image_url.endswith(".jpeg")):
            return image_url
    
    return None

@app.post("/summarize_person")
async def summarize_person(request: PersonRequest):
    search_results = search_person(request.name)
    if not search_results:
        raise HTTPException(status_code=404, detail="No information found")

    snippets = []
    for result in search_results:
        snippet = result.get("snippet", "")
        if isinstance(snippet, str):
            snippets.append(snippet)

    content = " ".join(snippets)

    if not content:
        raise HTTPException(status_code=404, detail="No content available to summarize")
    image_url = search_image(request.name)
    image_response = requests.get(image_url, stream=True, headers={"Accept": "image/png,image/jpeg"})

    person_summary = summarize_content(content)
    rupam_summary = get_rupam_bhattacharjee_data()

    comparison_prompt = f"""
    Compare the following two profiles and highlight their similarities, differences, and any notable points:

    Profile 1:
    {person_summary}

    Profile 2:
    {rupam_summary}
    """

    comparison_response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={"type": "text"},
        temperature=0.7,
        max_tokens=250,
        messages=[
            {"role": "system", "content": "You will be given two profiles. Compare them and provide the similarities, differences, and any notable points in 250 tokens."},
            {"role": "user", "content": comparison_prompt}
        ]
    )

    comparison_summary = comparison_response.choices[0].message.content


    if image_response.status_code != 200:
        raise HTTPException(status_code=image_response.status_code, detail="Failed to fetch image")

    headers = {"Comparative-analysis":comparison_summary}

    return StreamingResponse(image_response.iter_content(chunk_size=1024), media_type=image_response.headers.get("Content-Type"), headers=headers)

def get_rupam_bhattacharjee_data():
    data = """
    Rupam Bhattacharjee
    Empowering Businesses with Data-Driven Decisions | Using Data to Make the World a Better Place | Building UnoMiru VR 360 | Idea2MVP Grandmaster
    Singapore, Singapore

    Summary:
    Rupam is a passionate entrepreneur with a deep belief in the power of technology to change the world. As the Chairman & MD of WaysAhead Global Pte. Ltd., Rupam helps businesses innovate and grow through the use of deep tech. Rupam is also a co-founder and CEO of UnoMiru, a company that is developing cutting-edge VR 360 solutions. Rupam is a frequent public speaker and is involved in a number of professional organizations. Rupam is always looking for new opportunities to connect with like-minded individuals and help them achieve their goals.

    Experience:
    - Chairman & MD @ WaysAhead Global (3 years)
    - Co-Founder of WaysAhead Global (7 years)
    - Power BI Evangelist & Technical Consultant (Augmented Analytics) (9 years)
    - Deep-Tech Advisor & Strategist @ KalaaPlanet (7 months)
    - Deep-Tech Advisor & Strategist @ RenoSwift (7 months)
    - Co-Founder & CEO @ UnoMiru (1 year)
    - Visiting Faculty Member - Data Analytics @ PwC's Academy Middle East (5 years)
    - Corporate Training & Development (21 years)

    Skills:
    - Analytical Skills
    - Data Storytelling
    - Artificial Intelligence (AI)
    - Domain expertise in Robotics, AI, Retail, Fashion, Leisure & Entertainment, Cinemas, Healthcare, MEP, Structural, Mechanical, Urban Design & Infrastructure Mgmt & Finance.
    """
    return data
def create_wordcloud(text: str):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

@app.post("/compare_person")
async def compare_person(request: PersonRequest):
    search_results = search_person(request.name)
    if not search_results:
        raise HTTPException(status_code=404, detail="No information found")
    snippets = []
    for result in search_results:
        snippet = result.get("snippet", "")
        if isinstance(snippet, str):
            snippets.append(snippet)
    content = " ".join(snippets)
    if not content:
        raise HTTPException(status_code=404, detail="No content available to summarize")
    person_summary = summarize_content(content)
    rupam_summary = get_rupam_bhattacharjee_data()
    person_wordcloud = create_wordcloud(person_summary)
    rupam_wordcloud = create_wordcloud(rupam_summary)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(person_wordcloud, interpolation='bilinear')
    axes[0].set_title(request.name)
    axes[0].axis('off')    
    axes[1].imshow(rupam_wordcloud, interpolation='bilinear')
    axes[1].set_title("Rupam Bhattacharjee")
    axes[1].axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png" )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
