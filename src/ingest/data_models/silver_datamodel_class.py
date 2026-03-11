# ==========================
# Pivot Data Model  
# ==========================
import datetime

class Silver_Datamodel:
    def __init__(
            self, 
            id: str,
            source: str,
            url : str,
            title : str, 
            description : str,
            published_at : datetime,
            updated_at : datetime,
            status : str,
            rome_code : str,
            rome_label : str,
            title_description : str,
            contract_type : str,
            worktime : str,
            experience_level : str,
            experience_description : str,            
            naf_code : str,
            job_city: str,
            job_postal_code: str,
            company_name : str,             
            company_city : str,            
            company_postal_code : str,
            company_url : str,
            salary_min: float,
            salary_max: float,
            profile : str,          
            location_department: str,
            skills : list,
            salary_periodicity: str,
            unpublished_at : datetime = None            
 ):

        self.source = source
        self.id = id
        self.title = title
        self.description = description
        self.profile = profile  
        self.rome_code = rome_code
        self.rome_label = rome_label
        self.title_description = title_description
        self.contract_type = contract_type
        self.experience_level = experience_level
        self.salary_min = salary_min
        self.salary_max = salary_max
        self.job_city = job_city
        self.job_postal_code = job_postal_code
        self.location_department = location_department
        self.published_at = published_at
        self.updated_at = updated_at
        self.unpublished_at = unpublished_at
        self.url = url
        self.skills = skills
        self.company_name = company_name
        self.status = status
        self.worktime = worktime
        self.experience_description = experience_description
        self.naf_code = naf_code
        self.company_city = company_city
        self.company_postal_code = company_postal_code
        self.company_url = company_url
        self.salary_periodicity = salary_periodicity
        
    def __str__(self) -> str:
        """Surcharge affichage print()"""
        desc = self.description[:50] if self.description else "NA"  
        profile = self.profile[:50] if self.profile else "NA"
        return f"""- Id : #{self.id}
        title : {self.title}
        description : {desc}...
        profile : {profile}
        rome_code : {self.rome_code}
        rome_label : {self.rome_label}
        contract_type : {self.contract_type}
        experience_level : {self.experience_level}
        salary_min : {self.salary_min}
        salary_max : {self.salary_max}
        location_city : {self.location_city}
        location_department : {self.location_department}
        published_at : {self.published_at}
        updated_at : {self.updated_at}
        url : {self.url}
        skills : {self.skills}
        company_name : {self.company_name}"""
    
    def to_dict(self):
        """Convertit l'objet en dictionnaire pour DataFrame"""
        return {
            "id": self.id,
            "source": self.source,
            "url": self.url,
            "title": self.title,  
            "description": self.description,
            "published_at": self.published_at,
            "updated_at": self.updated_at,
            "unpublished_at": self.unpublished_at,
            "status": self.status,
            "rome_code": self.rome_code,
            "rome_label": self.rome_label,
            "title_description": self.title_description,
            "contract_type": self.contract_type,
            "worktime": self.worktime,
            "experience_level": self.experience_level,
            "experience_description": self.experience_description,
            "naf_code": self.naf_code,
            "job_city": self.job_city,
            "job_postal_code": self.job_postal_code,
            "company_name": self.company_name,
            "company_city": self.company_city,
            "company_postal_code": self.company_postal_code,
            "company_url": self.company_url,
            "salary_min": self.salary_min,
            "salary_max": self.salary_max,
            "profile": self.profile,
            "location_department": self.location_department,
            "skills": self.skills,
            "salary_periodicity": self.salary_periodicity
        }
    
    @classmethod
    def get_dataframe_columns(cls):
        """Returns the list of DataFrame column names (single source of truth)
        
        Extracts column names automatically from to_dict() to avoid duplication.
        """
        # Create a dummy instance to extract column names from to_dict()
        dummy = cls(
            id="", source="", url="",  title="", description="",
            published_at=None, updated_at=None, unpublished_at=None,
            status="", rome_code="", rome_label="", title_description="",
            contract_type="", worktime="", 
            experience_level="", experience_description="", naf_code="",
            job_city="", job_postal_code="",
            company_name="", company_city="", company_postal_code="", company_url="",
            salary_min=0.0, salary_max=0.0, 
            profile="",
            location_department="",
            skills=[], 
            salary_periodicity=""
        )
        return list(dummy.to_dict().keys())


# ==========================
# Class WTTJ 
# Before merging with Silver_Datamodel 
# Extracted data form json, clean and enrich with ROME code and label.
# ==========================
class silver_wttj:
    def __init__(
            self, 
            reference: str,
            name : str,
            description : str,
            profile : str,
            salary_min: float,
            salary_max: float,
            salary_currency: str,
            education_level: str,
            company_summary: str,
            company_description: str,
            updated_at : datetime,
            published_at : datetime,
            archived_at : datetime,
            contract_duration_min : str,
            remote:str,
            ats:str,
            contract_duration_max :str,
            experience_level : str,
            contract_type : str,
            urls:list = None,
            canonical_url:str = None,
            skills : list = None,
            key_missions : list = None,
            offices: list = None,
            sectors: list = None,
            profession : str = None 
 ):

        self.reference = reference
        self.name = name    
        self.description = description
        self.profile = profile 
        self.salary_min = salary_min
        self.salary_max = salary_max
        self.salary_currency = salary_currency
        self.education_level = education_level
        self.company_summary = company_summary
        self.company_description = company_description
        self.updated_at = updated_at
        self.published_at = published_at
        self.archived_at = archived_at
        self.contract_duration_min = contract_duration_min
        self.remote = remote
        self.ats = ats
        self.contract_duration_max = contract_duration_max
        self.experience_level = experience_level
        self.contract_type = contract_type
        self.urls = urls
        self.canonical_url = canonical_url
        self.skills = skills
        self.key_missions = key_missions
        self.offices = offices
        self.sectors = sectors
        self.profession = profession


    def __str__(self) -> str:
        """Surcharge affichage print()"""
        desc = self.description[:50] if self.description else "NA"
        profile = self.profile[:50] if self.profile else "NA"
        
        return f"""- Id : #{self.reference}
        job_title : {self.name}
        job_description : {desc}...
        company_employees : {self.company_employees}
        profile : {profile}
        url : {self.url}"""


class NormalizeWTTJResult:
    def __init__(self, job_id, status, dt, output_format, files, errors):
        self.job_id = job_id
        self.status = status
        self.dt = dt
        self.format = output_format
        self.files = files
        self.errors = errors
