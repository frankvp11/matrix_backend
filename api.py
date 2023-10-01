# main.py
from fastapi import FastAPI, UploadFile, HTTPException
import torch
from torchvision import transforms
import cv2
from ultralytics import YOLO

import numpy as np
app = FastAPI()
from PIL import Image
from io import BytesIO

# Load the YOLOv8 model and its weights
model = YOLO("best2.pt")
# model.eval()

def sort_bboxes( bboxes):
    # Calculate the center y-coordinate of each bounding box
        bboxes = sorted(bboxes, key=lambda x: ((x[1] + x[3]) / 2, x[0]))

        # Determine the average height of bounding boxes to use as a heuristic
        avg_height = sum([box[3] - box[1] for box in bboxes]) / len(bboxes)

        sorted_bboxes = []
        row = []
        for i in range(len(bboxes)):
            row.append(bboxes[i])

            if i == len(bboxes) - 1 or bboxes[i+1][1] - bboxes[i][3] > avg_height / 2:
                row = sorted(row, key=lambda x: x[0])  # sort the row based on x
                sorted_bboxes.extend(row)
                row = []

        return sorted_bboxes    

def form_matrix( bboxes, N=3, M=3):
        # Sort by y value to group by rows
        def group_by_rows(bboxes):
            bboxes = sorted(bboxes, key=lambda x: x[1])
            avg_height = sum([box[3] - box[1] for box in bboxes]) / len(bboxes)

            # Group into rows
            rows = []
            row = [bboxes[0]]
            for i in range(1, len(bboxes)):
                if bboxes[i][1] > row[-1][1] + (avg_height * 0.5):  # adjustable factor
                    rows.append(sorted(row, key=lambda x : x[0]))
                    row = []
                row.append(bboxes[i])
            rows.append(sorted(row, key=lambda x : x[0]))  # Add the last row
            return rows

        # Group each row into columns using x-coordinates
        def group_by_columns(bboxes):
            bboxes_sorted = sorted(bboxes, key=lambda x: x[0])
            avg_width = sum([box[2] - box[0] for box in bboxes_sorted]) / len(bboxes_sorted)

            cols = []
            col = [bboxes_sorted[0]]
            temp = [box[-1] for box in bboxes_sorted]
            for i in range(1, len(bboxes_sorted)):
                # Calculate the gap between the current bounding box and the previous one
                gap =  bboxes_sorted[i][0] -col[-1][0] 
                # If the gap is larger than half the average width, create a new column
                if abs(gap) > (avg_width * 0.5):  # adjustable factor

                    cols.append(sorted(col, key=lambda x : x[1]))
                    col = []
                
                col.append(bboxes_sorted[i])

            # Append the last column
            if col:
                cols.append(sorted(col, key=lambda x : x[1]))

            return cols

        rows = group_by_rows(bboxes)
        new_rows =[]
        for row in rows:
            new_rows.append([i[-1] for i in row])
        rows = new_rows
        print(f"Rows {new_rows}")
        cols = group_by_columns(bboxes)
        new_cols = []
        for col in cols:
            new_cols.append([i[-1] for i in col])
        cols = new_cols
        print(f"Columns {cols}")

        def merge_matrices(rows_matrix, cols_matrix):
            # Calculate dimensions
            num_rows = len(rows_matrix)

            num_cols = max(max(len(row) for row in rows_matrix), max(len(col) for col in cols_matrix))

            merged_matrix = []

            # Iterate over each cell by row and column
            for i in range(num_rows):
                row = []
                for j in range(num_cols):
                    if len(rows_matrix) > i and len(rows_matrix[i]) > j:
                        row_value = rows_matrix[i][j]
                    else:
                        row_value = ""

                    if len(cols_matrix) > j and len(cols_matrix[j]) > i:
                        col_value = cols_matrix[j][i]
                    else:
                        col_value = ""

                    # Choose value from either rows_matrix or cols_matrix. If both have a value, they should match
                    if row_value and col_value and row_value != col_value:
                        print(f"Conflicting values at {i}, {j}: {row_value} vs. {col_value}")
                        row.append(" ")
                    else:
                        value = row_value or col_value
                        row.append(value)
                merged_matrix.append(row)

            return merged_matrix
        return merge_matrices(rows, cols)

@app.get("/")
async def temp():
     return {"hi":"Frank"}

@app.post("/predict/")
async def predict_image(file: UploadFile = None):
    if not file:
        raise HTTPException(status_code=400, detail="No file received.")

    try:
        file_bytes = np.asarray(bytearray(await file.read()), dtype=np.uint8)

        cv2_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        cv2.imwrite("original.jpg", cv2_image)

        results = model.predict(cv2_image, conf=0.4)
        for result in results:
                    
                    for (x0, y0, x1, y1), (cls) in zip(result.boxes.xyxy, result.boxes.cls):
                        
                        cv2.rectangle(cv2_image, (int(x0), int(y0)), (int(x1), int(y1)), color=(0,255,0), thickness=2)
                        cv2.putText(cv2_image, str(cls.item()), (int(x0), int(y0)-5), fontFace = cv2.FONT_ITALIC, fontScale = 0.6, color = (0, 255, 0), thickness=2)

        list_of_coords =[ ]
        for box in results:
            for (x0, y0, x1, y1), (cls) in zip(box.boxes.xyxy, box.boxes.cls):
                list_of_coords.append([x0.item(), y0.item(), x1.item(), y1.item(), cls.item()])
        try:
                    sorted_boxes = (sort_bboxes(list_of_coords))
                    matrix = form_matrix(sorted_boxes, 4, 3)
        except Exception as e:
                    print(e)
                    matrix = [[1,2,3], [4,5,6], [7,8,9]]

        # print(matrix)
        for row in matrix:
             print(row)
        cv2.imwrite("pred.jpg", cv2_image)
        return {"predictions": matrix}


    except Exception as e:
        print(f"Exception type: {type(e)} - Exception message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
