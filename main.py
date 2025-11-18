import uvicorn
import uuid
import pipeline  # Your custom ML module
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, Depends
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
# Import new types
from sqlalchemy import Column, String, select, Float, Integer, ForeignKey
from sqlalchemy.orm import relationship # Import relationship
from typing import List # Import List for response models

# --- Database Configuration ---
DATABASE_URL = "sqlite+aiosqlite:///./jobs.db"
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# --- Database Models ---
class Job(Base):
    """Stores job descriptions."""
    __tablename__ = "jobs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    text = Column(String, nullable=False)
    
    # Add a relationship so we can easily get candidates
    candidates = relationship("Candidate", back_populates="job")

# --- NEW CANDIDATE MODEL ---
class Candidate(Base):
    """Stores candidate info and their score."""
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    
    # Foreign key to link to the Job
    job_id = Column(String, ForeignKey("jobs.id"))
    
    # Add a relationship back to the Job
    job = relationship("Job", back_populates="candidates")

# --- Pydantic Models (for response validation) ---
# We need these so FastAPI knows how to format the JSON
from pydantic import BaseModel

class CandidateOut(BaseModel):
    id: int
    full_name: str
    email: str
    score: float
    
    class Config:
        from_attributes = True # Read data from SQLAlchemy models

class JobOut(BaseModel):
    id: str
    text: str
    
    class Config:
        from_attributes = True


# --- Dependency ---
async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session

# --- FastAPI App ---
app = FastAPI(
    title="Resume Scorer API",
    description="An API for scoring resumes against job descriptions.",
    version="2.0.0"
)

@app.on_event("startup")
async def startup():
    """Create all database tables on startup."""
    async with engine.begin() as conn:
        # This will now create both 'jobs' and 'candidates' tables
        await conn.run_sync(Base.metadata.create_all)

# --- Frontend Endpoints ---
@app.get("/")
async def read_root():
    """Serves the student portal (index.html)."""
    return FileResponse('index.html')

@app.get("/recruiter")
async def read_recruiter_page():
    """Serves the recruiter job creation page (recruiter.html)."""
    return FileResponse('recruiter.html')

# --- NEW DASHBOARD PAGE ENDPOINT ---
@app.get("/dashboard")
async def read_dashboard_page():
    """Serves the new recruiter dashboard (dashboard.html)."""
    return FileResponse('dashboard.html')


# --- API Endpoints ---

@app.post("/create-job/")
async def create_job(
    job_description: str = Form(..., description="The full text of the job description"),
    session: AsyncSession = Depends(get_session)
):
    try:
        new_job = Job(text=job_description)
        session.add(new_job)
        await session.commit()
        await session.refresh(new_job)
        return {"job_id": new_job.id}
    except Exception as e:
        await session.rollback()
        print(f"Error creating job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job in database: {e}"
        )

@app.get("/job/{job_id}")
async def get_job(job_id: str, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found."
        )
    return {"text": job.text}


# --- NEW ENDPOINTS FOR DASHBOARD ---

@app.get("/jobs/", response_model=List[JobOut])
async def get_all_jobs(session: AsyncSession = Depends(get_session)):
    """Fetches all jobs for the dashboard."""
    result = await session.execute(select(Job))
    jobs = result.scalars().all()
    return jobs

@app.get("/job/{job_id}/candidates", response_model=List[CandidateOut])
async def get_job_candidates(job_id: str, session: AsyncSession = Depends(get_session)):
    """Fetches all candidates and scores for a specific job."""
    result = await session.execute(
        select(Candidate).where(Candidate.job_id == job_id).order_by(Candidate.score.desc())
    )
    candidates = result.scalars().all()
    if not candidates:
        return [] # Return empty list, not an error
    return candidates


# --- UPDATED SCORING ENDPOINT ---
@app.post("/score-resume/")
async def calculate_score(
    # --- ADD NEW FORM FIELDS ---
    full_name: str = Form(..., description="The candidate's full name"),
    email: str = Form(..., description="The candidate's email address"),
    # ---
    resume_file: UploadFile = File(..., description="The user's resume file"),
    job_id: str = Form(..., description="The unique ID of the job to score against"),
    session: AsyncSession = Depends(get_session)
):
    # 1. Fetch the Job Description
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    jd_text = job.text

    # 2. Check for valid resume file types
    allowed_types = [
        'application/pdf', 
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ]
    if resume_file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .pdf or .docx file.")

    try:
        # 3. Extract text from the resume
        resume_text = pipeline.extract_text_from_file(
            file=resume_file.file, 
            content_type=resume_file.content_type
        )
        if not resume_text.strip():
            raise HTTPException(status_code=422, detail="Could not extract text from resume.")

        # 4. Run the ML pipeline
        score = pipeline.get_similarity_score(resume_text, jd_text)
        match_percentage = round(score * 100, 2)
        
        # --- 5. NEW: Save the candidate to the database ---
        new_candidate = Candidate(
            full_name=full_name,
            email=email,
            score=match_percentage,
            job_id=job_id
        )
        session.add(new_candidate)
        await session.commit()
        # -----------------------------------------------
        
        return {"match_percentage": match_percentage}

    except HTTPException as e:
        await session.rollback()
        raise e
    except Exception as e:
        await session.rollback()
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}"
        )

# --- Run the Server ---
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )