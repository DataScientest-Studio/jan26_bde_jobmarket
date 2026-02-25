# ==========================
# Pivot Data Model  
# ==========================
import datetime
from src.ingest.silver.welcome_to_jungle import get_field_or_default

class Silver_Datamodel:
    def __init__(
            self, 
            source: str,
            id: str,
            title : str, 
            description : str,
            profile : str,          
            rome_code : str,
            rome_label : str,
            contract_type : str,
            experience_level : str,
            salary_min: float,
            salary_max: float,
            location_city: str,
            location_department: str,
            published_at : datetime,
            updated_at : datetime,
            url : str,
            existing_skills : list,
            company_name : str 
            
 ):

        self.source = source
        self.id = id
        self.title = title
        self.description = description
        self.profile = profile  
        self.rome_code = rome_code
        self.rome_label = rome_label
        self.contract_type = contract_type
        self.experience_level = experience_level
        self.salary_min = salary_min
        self.salary_max = salary_max
        self.location_city = location_city
        self.location_department = location_department
        self.published_at = published_at
        self.updated_at = updated_at
        self.url = url
        self.existing_skills = existing_skills
        self.company_name = company_name
        
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
        existing_skills : {self.existing_skills}
        company_name : {self.company_name}"""
    
    def to_dict(self):
        """Convertit l'objet en dictionnaire pour DataFrame"""
        return {
            "source": self.source,
            "id": self.id,
            "intitule": self.title,  # Mapping title → intitule
            "description": self.description,
            "profile": self.profile,
            "rome_code": self.rome_code,
            "rome_label": self.rome_label,
            "contract_type": self.contract_type,
            "experience_level": self.experience_level,
            "salary_min": self.salary_min,
            "salary_max": self.salary_max,
            "location_city": self.location_city,
            "location_department": self.location_department,
            "published_at": self.published_at,
            "updated_at": self.updated_at,
            "url": self.url,
            "existing_skills": self.existing_skills,
            "company_name": self.company_name
        }
    
    @classmethod
    def get_dataframe_columns(cls):
        """Returns the list of DataFrame column names (single source of truth)
        
        Extracts column names automatically from to_dict() to avoid duplication.
        """
        # Create a dummy instance to extract column names from to_dict()
        dummy = cls(
            source="", id="", title="", description="", profile="",
            rome_code="", rome_label="", contract_type="", experience_level="",
            salary_min=None, salary_max=None, location_city="", location_department="",
            published_at=None, updated_at=None, url="", existing_skills=[], company_name=""
        )
        return list(dummy.to_dict().keys())