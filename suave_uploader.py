# suave_uploader.py

import io
import requests

def upload_to_suave(df, survey_name, user, password, referer, dzc_file=None, csv_filename=None, views=None):
    """
    Upload a DataFrame to SuAVE as a new survey.
    `csv_filename` (the originating survey's stored CSV name, e.g. "user_Survey.csv")
    is used to look up that survey's view list so the new survey inherits it —
    pass `views` directly instead if you already have it and want to skip that
    lookup.
    Returns: (success_flag, message, new_url)
    """
    try:
        # Step 1: Log in with session
        s = requests.Session()
        headers = {
            "User-Agent": "suave user agent",
            "referer": referer
        }

        login_response = s.post(
            referer + "login",
            headers=headers,
            data={
                "user": user,
                "pass": password,
                "remember-me": "true"
            }
        )

        if login_response.status_code != 200:
            return (False, f"Login failed (status {login_response.status_code})", None)

        # Step 2: Inherit the originating survey's views, so the new survey
        # doesn't lose tabs (like Jupyter or Streamlit) that aren't in the
        # server's default view list.
        if views is None and csv_filename:
            try:
                orig_survey = (
                    csv_filename[len(user) + 1:-4]
                    if csv_filename.startswith(user + "_")
                    else csv_filename[:-4]
                )
                dzc_resp = s.get(
                    referer + "getSurveyDzc",
                    params={"user": user, "file": orig_survey},
                    headers=headers,
                )
                if dzc_resp.status_code == 200:
                    views = dzc_resp.json().get("views")
            except Exception:
                pass  # non-fatal — upload proceeds with the server's default views

        # Step 3: Prepare the CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        files = {
            "file": (f"{survey_name}.csv", csv_buffer.getvalue())
        }

        data = {
            "name": survey_name,
            "user": user
        }

        if dzc_file:
            data["dzc"] = dzc_file
        if views:
            data["views"] = ",".join(views) if isinstance(views, list) else views

        upload_url = referer + "uploadSurvey"
        upload_response = s.post(upload_url, files=files, data=data, headers=headers)

        if upload_response.status_code == 200:
            new_survey_url = f"{referer}?survey={user}_{survey_name}.csv"
            return (True, "✅ Survey uploaded successfully!", new_survey_url)
        else:
            return (False, f"Upload failed ({upload_response.status_code}) – {upload_response.text}", None)

    except Exception as e:
        return (False, f"Upload failed due to error: {e}", None)
