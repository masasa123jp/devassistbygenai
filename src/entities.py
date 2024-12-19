import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EmailAddress:
    email: str

    def __post_init__(self):
        if not self._validate_email(self.email):
            raise ValueError('Invalid email format!')

    @staticmethod
    def _validate_email(email: str):
        pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        if re.match(pattern, email):
            return True
        return False


@dataclass(frozen=True)
class Project:
    key: str
    name_project: str

@dataclass(frozen=True)
class File:
    name: str
    size: str
    path: str
    value:str

@dataclass(frozen=True)
class Issue:
    key: str
    name_issue: str
    task_description: str
    status: str
    assignee: str

    def __post_init__(self):
        if not self._validate_issue_id(self.key):
            raise ValueError("Invalid issue id format!")

    @staticmethod
    def _validate_issue_id(issue_id: str):
        # The issue_id usually looks like "XYZ-1". Modify this regex for your needs
        #print("issue_id: " + issue_id)
        pattern = r'[A-Z]+[0-9]+-\d+'
        if re.match(pattern, issue_id):
            return True
        return False
