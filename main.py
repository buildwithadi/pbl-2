import uvicorn
import uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, Depends
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, select
import pipeline  # Our custom ML module

# --- Database Configuration ---
DATABASE_URL = "sqlite+aiosqlite:///./jobs.db"  # This will create a file named jobs.db
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# --- Database Model ---
class Job(Base):
    """
    Database model for storing job descriptions.
    We use a UUID string as the primary key for shareable, non-guessable links.
    """
    __tablename__ = "jobs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    text = Column(String, nullable=False)

# --- Dependency to get DB session ---
async def get_session() -> AsyncSession:
    """Dependency to provide a database session to API endpoints."""
    async with async_session() as session:
        yield session

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Resume Scorer API",
    description="An API for scoring resumes against job descriptions.",
    version="2.0.0"
)

@app.on_event("startup")
async def startup():
    """Create the database tables on application startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# --- Frontend Endpoint ---
@app.get("/")
async def read_root():
    """Serves the main index.html file as the root page."""
    return FileResponse('index.html')

# --- API Endpoints ---

@app.post("/create-job/")
async def create_job(
    job_description: str = Form(..., description="The full text of the job description"),
    session: AsyncSession = Depends(get_session)
):
    """
    Recruiter-facing endpoint.
    Creates a new job description in the database and returns its unique ID.
    """
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
    """
    Student-facing endpoint.
    Fetches the text of a job description by its ID.
    """
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found. This link may be invalid or expired."
        )
    return {"text": job.text}


@app.post("/score-resume/")
async def calculate_score(
    resume_file: UploadFile = File(..., description="The user's resume file"),
    job_id: str = Form(..., description="The unique ID of the job to score against"),
    session: AsyncSession = Depends(get_session)
):
    """
    Student-facing endpoint.
    Calculates the score by fetching the job from the DB using job_id.
    """
    
    # 1. Fetch the Job Description from the database
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found. Cannot score."
        )
    jd_text = job.text

    # 2. Check for valid resume file types
    allowed_types = [
        'application/pdf', 
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ]
    if resume_file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload a .pdf or .docx file."
        )

    try:
        # 3. Extract text from the resume
        resume_text = pipeline.extract_text_from_file(
            file=resume_file.file, 
            content_type=resume_file.content_type
        )
        if not resume_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract text from resume. File might be empty."
            )

        # 4. Run the ML pipeline
        score = pipeline.get_similarity_score(resume_text, jd_text)
        match_percentage = round(score * 100, 2)
        
        return {"match_percentage": match_percentage}

    except HTTPException as e:
        raise e  # Re-raise known exceptions
    except Exception as e:
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