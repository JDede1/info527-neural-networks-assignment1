# INFO 527 — Neural Networks  
## Assignment 1: String-to-Vectors (stv_nn)

### Objectives
The objectives of this assignment are to:
1. Practice Python programming and use of NumPy arrays.
2. Understand the importance of testing code for correctness and reliability.
3. Implement an `Index` class that maps string or object sequences to numeric vectors and matrices, a common preprocessing step in neural network training.

---

### Files in This Repository
| File | Description |
|------|--------------|
| `notebooks/stv_nn.ipynb` | Main notebook containing the full implementation of the `Index` class. |
| `notebooks/string-to-vector.ipynb` | Notebook used to run pytest validation and record test results. |
| `tests/test_nn.py` | Provided test file used to validate correctness using pytest. |
| `stv_nn.py` | The exported version of the main notebook used for testing. |
| `stv_nn.html` | Exported HTML version of the notebook for grading and review. |
| `README.md` | This file, describing objectives, setup, and instructions. |
| `requirements.txt` | Python dependencies for running tests. |

---

### Class Overview: `Index`
The `Index` class provides a numeric mapping for sequences of objects (typically strings). It supports:
- Creating object-to-index and index-to-object mappings.
- Converting between object sequences and numeric arrays.
- Generating binary (one-hot) encodings and decoding them back to objects.
- Handling padding and unknown objects gracefully.

**Implemented Methods:**
1. `__init__` – Assign indexes to unique vocabulary items starting from a given value.
2. `objects_to_indexes` – Convert a sequence of objects to a numeric index vector.
3. `objects_to_index_matrix` – Convert sequences of objects to an indexed matrix with padding.
4. `objects_to_binary_vector` – Produce a binary (one-hot) vector representation.
5. `objects_to_binary_matrix` – Produce a binary matrix for multiple sequences.
6. `indexes_to_objects` – Convert indexes back to their original objects.
7. `index_matrix_to_objects` – Convert index matrices back to object sequences.
8. `binary_vector_to_objects` – Retrieve objects from a binary vector.
9. `binary_matrix_to_objects` – Retrieve objects from a binary matrix.

---

### Environment Setup
This assignment reuses the shared virtual environment created for previous coursework.

```bash
# Activate the shared environment
source ~/venvs/ml-env/bin/activate

# Navigate to the assignment folder
cd ~/projects/info527-neural-networks-assignment1

# Install dependencies
pip install -r requirements.txt
````

**requirements.txt**

```
numpy
pytest
```

---

### Testing Instructions

1. Run all cells in `stv_nn.ipynb` to complete the implementation.
2. Export the notebook as:

   * `stv_nn.ipynb`
   * `stv_nn.html`
   * `stv_nn.py`
3. Place the exported `.py` file in the same directory as `test_nn.py`.
4. Run pytest to verify correctness:

```bash
pytest tests/test_nn.py
```

**Expected Output:**

```
============================= test session starts ==============================
collected 8 items

test_nn.py ........                                                      [100%]

============================== 8 passed in 1.XXs ===============================
```

---

### Submission Files

Submit the following for grading:

* `stv_nn.ipynb`
* `stv_nn.html`
* `stv_nn.py`
* `string-to-vector.ipynb` (with pytest results)

---

### Grading Criteria

Grading is based solely on the correctness of the code written in `stv_nn.ipynb`.

| Method                     | Description                         | Marks |
| -------------------------- | ----------------------------------- | ----- |
| `__init__`                 | Index initialization                | 3     |
| `objects_to_indexes`       | Convert objects to index vector     | 1     |
| `objects_to_index_matrix`  | Convert sequences to indexed matrix | 3     |
| `objects_to_binary_vector` | One-hot vector encoding             | 2     |
| `objects_to_binary_matrix` | One-hot matrix encoding             | 1     |
| `indexes_to_objects`       | Decode index vector                 | 1     |
| `index_matrix_to_objects`  | Decode index matrix                 | 1     |
| `binary_vector_to_objects` | Decode binary vector                | 2     |
| `binary_matrix_to_objects` | Decode binary matrix                | 1     |

**Total: 15 marks**

---

### Author

This repository was completed as part of INFO 527: Neural Networks, under the M.S. in Information Science and Machine Learning program (2023–2025).

```

---
