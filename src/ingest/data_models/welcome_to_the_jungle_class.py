# ==========================
# Class WTTJ
# ==========================
# TODO : remove in favor of dataclass ?
import datetime
from src.utils.wttj_utils import get_field_or_default


class WTTJ:
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
        desc = self.job_description[:50] if self.job_description else "NA"
        profile = self.profile[:50] if self.profile else "NA"
        
        return f"""- Id : #{self.reference}
        job_title : {self.name}
        job_description : {desc}...
        company_employees : {self.company_employees}
        profile : {profile}
        url : {self.url}"""
