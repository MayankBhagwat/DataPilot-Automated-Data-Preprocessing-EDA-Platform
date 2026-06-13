import math
import traceback
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.cleaning import clean_dataset
from app.eda import dataset_info
from app.preprocessing import preprocess_dataset


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
PROCESSED_FOLDER = BASE_DIR / "processed"
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

UPLOAD_FOLDER.mkdir(exist_ok=True)
PROCESSED_FOLDER.mkdir(exist_ok=True)

app = FastAPI(title="Remora AI", version="1.1.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def sanitize_data(data):
    if isinstance(data, dict):
        return {str(key): sanitize_data(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [sanitize_data(value) for value in data]
    if isinstance(data, (datetime, date, pd.Timestamp)):
        return data.isoformat()
    if isinstance(data, (np.bool_, bool)):
        return bool(data)
    if isinstance(data, (np.integer,)):
        return int(data)
    if isinstance(data, (np.floating, float)):
        value = float(data)
        return value if math.isfinite(value) else None
    if data is pd.NA or data is None:
        return None
    return data


def _save_upload(upload: UploadFile, destination: Path) -> None:
    total = 0
    with destination.open("wb") as output:
        while chunk := upload.file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise ValueError("File is larger than the 50 MB upload limit.")
            output.write(chunk)


def _read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_excel(path)


def _read_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        columns = pd.read_csv(path, nrows=0).columns
    else:
        columns = pd.read_excel(path, nrows=0).columns
    return [str(column) for column in columns]


def _auto_target(columns: list[object]) -> str | None:
    by_lower_name = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in (
        "target",
        "label",
        "outcome",
        "charges",
        "price",
        "sale_price",
        "claim_amount",
    ):
        if candidate in by_lower_name:
            return by_lower_name[candidate]
    return None


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/columns/")
def inspect_columns(file: UploadFile = File(...)):
    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Unsupported file type."},
        )

    upload_path = UPLOAD_FOLDER / f"{uuid4().hex}{extension}"
    try:
        _save_upload(file, upload_path)
        columns = _read_columns(upload_path)
        return {
            "status": "success",
            "columns": columns,
            "suggested_target": _auto_target(columns),
        }
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Could not read dataset columns."},
        )
    finally:
        file.file.close()
        upload_path.unlink(missing_ok=True)


@app.post("/upload/")
def upload_file(
    file: UploadFile = File(...),
    target_column: str | None = Form(default=None),
):
    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Unsupported file type. Upload a CSV, XLSX, or XLS file.",
            },
        )

    upload_path = UPLOAD_FOLDER / f"{uuid4().hex}{extension}"
    try:
        _save_upload(file, upload_path)
        df = _read_dataset(upload_path)
        if df.empty or df.shape[1] == 0:
            raise ValueError("The uploaded dataset is empty.")

        raw_eda_data = dataset_info(df)
        cleaned_df, cleaning_report = clean_dataset(df, return_report=True)
        selected_target = (target_column or "").strip() or _auto_target(cleaned_df.columns.tolist())
        processed_df, preprocessing_report = preprocess_dataset(
            cleaned_df,
            target_column=selected_target,
            return_report=True,
        )

        output_id = uuid4().hex[:8]
        cleaned_name = f"cleaned_{Path(original_name).stem}_{output_id}.csv"
        model_name = f"model_ready_{Path(original_name).stem}_{output_id}.csv"
        cleaned_df.to_csv(PROCESSED_FOLDER / cleaned_name, index=False)
        processed_df.to_csv(PROCESSED_FOLDER / model_name, index=False)

        return {
            "status": "success",
            "filename": cleaned_name,
            "cleaned_filename": cleaned_name,
            "model_filename": model_name,
            "results": {
                "eda": sanitize_data(raw_eda_data),
                "cleaning": {
                    "status": "Completed",
                    "duplicates_found": cleaning_report["duplicates_removed"],
                    **cleaning_report,
                },
                "preprocessing": {
                    "status": "Completed",
                    **preprocessing_report,
                    "new_shape": list(processed_df.shape),
                },
            },
        }
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(exc)},
        )
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "The dataset could not be processed.",
            },
        )
    finally:
        file.file.close()
        upload_path.unlink(missing_ok=True)


@app.get("/download/{filename}")
def download(filename: str):
    safe_name = Path(filename).name
    if safe_name != filename:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Invalid filename."},
        )

    target_path = PROCESSED_FOLDER / safe_name
    if target_path.is_file():
        return FileResponse(
            path=target_path,
            filename=safe_name,
            media_type="text/csv",
        )

    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "File not found."},
    )
