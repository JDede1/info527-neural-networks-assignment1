#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
The main code for the Strings-to-Vectors assignment. See README.md and Instructions for details.
"""
from typing import Sequence, Any

import numpy as np


class Index:
    """
    Represents a mapping from a vocabulary (e.g., strings) to integers.
    """

    def __init__(self, vocab: Sequence[Any], start=0):
        """
        Assigns an index to each unique item in the `vocab` iterable,
        with indexes starting from `start`.

        Indexes should be assigned in order, so that the first unique item in
        `vocab` has the index `start`, the second unique item has the index
        `start + 1`, etc.
        """
        # Initialize the dictionaries for object-to-index and index-to-object mappings
        self.start = start  # Store the start attribute
        self.object_to_index = {}
        self.index_to_object = {}

        # Initialize the starting index
        current_index = start

        # Assign an index to each unique item in the vocabulary
        for item in vocab:
            if item not in self.object_to_index:  # Only assign if item is not already indexed
                self.object_to_index[item] = current_index
                self.index_to_object[current_index] = item
                current_index += 1  # Increment the index for the next unique item
                

    def objects_to_indexes(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a vector of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array of the object indexes.
        """
        # Convert each object to its corresponding index using list comprehension
        # If the object is not found in the vocabulary, return `start - 1`
        indexes = [self.object_to_index.get(obj, self.start - 1) for obj in object_seq]
        
        return np.array(indexes)


    def objects_to_index_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a matrix of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        If the sequences are not all of the same length, shorter sequences will
        have padding added at the end, with `start-1` used as the pad value.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array of the object indexes.
        """
 # Find the maximum length of the sequences in object_seq_seq
        max_length = max(len(seq) for seq in object_seq_seq)
        # Initialize a matrix filled with `start-1` for padding
        matrix = np.full((len(object_seq_seq), max_length), self.start - 1)

        # Fill the matrix with the indexes of the objects
        for i, object_seq in enumerate(object_seq_seq):
            indexes = [self.object_to_index.get(obj, self.start - 1) for obj in object_seq]
            matrix[i, :len(indexes)] = indexes

        return matrix    
        

    def objects_to_binary_vector(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a binary vector, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """
# Initialize a binary vector of zeros with a length equal to the size of the vocabulary plus start
        binary_vector = np.zeros(len(self.object_to_index) + self.start, dtype=int)

        # Iterate through each object in the input sequence
        for obj in object_seq:
            if obj in self.object_to_index:
                index = self.object_to_index[obj]
                binary_vector[index] = 1

        return binary_vector
    

    def objects_to_binary_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a binary matrix, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array, where each row in the array corresponds
                 to a row in the input, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """
        # Calculate the number of columns needed, accounting for 'start'
        max_index = len(self.object_to_index) + self.start
        # Initialize a binary matrix of zeros
        binary_matrix = np.zeros((len(object_seq_seq), max_index), dtype=int)

        # Iterate through each sequence of objects
        for i, object_seq in enumerate(object_seq_seq):
            for obj in object_seq:
                if obj in self.object_to_index:
                    index = self.object_to_index[obj]
                    binary_matrix[i, index] = 1

        return binary_matrix


    def indexes_to_objects(self, index_vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of objects associated with the indexes in the input
        vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_vector: A 1-dimensional array of indexes
        :return: A sequence of objects, one for each index.
        """
        # Use list comprehension to retrieve objects for each index in index_vector.
        # Only include the object if the index exists in the index_to_object mapping.
        objects = [self.index_to_object[index] for index in index_vector if index in self.index_to_object]

        # Return the list of objects corresponding to the valid indexes.
        return objects

    

    def index_matrix_to_objects(
            self, index_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects associated with the indexes
        in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_matrix: A 2-dimensional array of indexes
        :return: A sequence of sequences of objects, one for each index.
        """
        # Initialize an empty list to store the sequences of objects
        objects_matrix = []

        # Iterate through each row (index vector) in the index matrix
        for index_vector in index_matrix:
            # Convert each row (1D array of indexes) to a sequence of objects
            # Use list comprehension to retrieve objects for each valid index in index_vector
            objects = [self.index_to_object[index] for index in index_vector if index in self.index_to_object]
    
            # Append the sequence of objects to the result list (objects_matrix)
            objects_matrix.append(objects)

        # Return the list of sequences
        return objects_matrix

    

    def binary_vector_to_objects(self, vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of the objects identified by the nonzero indexes in
        the input vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param vector: A 1-dimensional binary array
        :return: A sequence of objects, one for each nonzero index.
        """
        # Find the indexes of the nonzero elements in the vector.
        # [0] is used to get the first array (for the nonzero indexes).
        nonzero_indexes = np.nonzero(vector)[0]

        # Use list comprehension to retrieve the objects associated with the nonzero indexes.
        # Only include the object if the index exists in the index_to_object mapping.
        objects = [self.index_to_object[index] for index in nonzero_indexes if index in self.index_to_object]

        # Return the list of objects.
        return objects

    

    def binary_matrix_to_objects(
            self, binary_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects identified by the nonzero
        indices in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param binary_matrix: A 2-dimensional binary array
        :return: A sequence of sequences of objects, one for each nonzero index.
        """
        # Initialize an empty list to store the sequences of objects for each binary vector (row) in the binary matrix
        objects_matrix = []

        # Iterate through each row (binary vector) in the binary matrix
        for binary_vector in binary_matrix:
            nonzero_indexes = np.nonzero(binary_vector)[0]
    
            # Use list comprehension to retrieve the objects associated with the nonzero indexes
            # Only include the object if the index exists in the index_to_object mapping
            objects = [self.index_to_object[index] for index in nonzero_indexes if index in self.index_to_object]
    
            # Append the sequence of objects corresponding to the current binary vector to the objects_matrix
            objects_matrix.append(objects)

        # Return the list of sequences
        return objects_matrix

