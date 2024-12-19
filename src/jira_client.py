# -*- coding: utf-8 -*-
from jira import JIRA
from src.entities import Project, Issue
from functools import wraps
import streamlit as st
from src.config import JIRA_CONFIG
from src.error_handler import handle_api_errors

@handle_api_errors
def create_jira_client():
    return JIRA(server=JIRA_CONFIG['server'], basic_auth=(JIRA_CONFIG['user_name'], JIRA_CONFIG['api_token']))

@handle_api_errors
def get_all_projects(jira_client) -> list:
    return [Project(project.key, project.name) for project in jira_client.projects()]

@handle_api_errors
def get_user_backlog(jira_client, project_key, user_account_id) -> list:
    query = f"project='{project_key}'"
    if user_account_id:
        query += f" AND assignee='{user_account_id}'"
    issues = jira_client.search_issues(query, maxResults=False)
    return [Issue(issue.key, issue.fields.summary, 
                  issue.fields.description or 'なし', 
                  issue.fields.status.name, 
                  getattr(issue.fields.assignee, 'displayName', '未割り当て')) for issue in issues]
