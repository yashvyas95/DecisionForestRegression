"""
Node class for decision tree implementation.

This module contains the Node class that represents individual nodes
in a decision tree, including both internal nodes and leaf nodes.
"""

from typing import Optional, Any, Dict, List
import numpy as np


class Node:
    """
    Represents a node in a decision tree.
    
    A node can be either an internal node (with left and right children)
    or a leaf node (with a prediction value). Internal nodes contain
    splitting criteria, while leaf nodes contain the final prediction.
    
    Attributes:
        feature: Index of the feature used for splitting (internal nodes only)
        threshold: Threshold value for the split (internal nodes only)
        value: Prediction value (leaf nodes only) or node statistics
        left: Left child node
        right: Right child node
        depth: Depth of this node in the tree
        n_samples: Number of samples that reached this node
        mse: Mean squared error at this node
        is_leaf: Whether this is a leaf node
    """
    
    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        value: Optional[float] = None,
        left: Optional['Node'] = None,
        right: Optional['Node'] = None,
        depth: int = 0,
        n_samples: int = 0,
        mse: float = 0.0
    ) -> None:
        """
        Initialize a new node.
        
        Args:
            feature: Feature index for splitting (internal nodes)
            threshold: Splitting threshold (internal nodes)
            value: Prediction value (leaf nodes)
            left: Left child node
            right: Right child node
            depth: Depth of the node in the tree
            n_samples: Number of samples at this node
            mse: Mean squared error at this node
        """
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        self.depth = depth
        self.n_samples = n_samples
        self.mse = mse
        self.is_leaf = feature is None
    
    def is_leaf_node(self) -> bool:
        """
        Check if this node is a leaf node.
        
        Returns:
            True if this is a leaf node, False otherwise
        """
        return self.is_leaf
    
    def predict(self, X: np.ndarray) -> float:
        """
        Make a prediction for a single sample.
        
        Args:
            X: Input sample as 1D numpy array
            
        Returns:
            Predicted value
            
        Raises:
            ValueError: If the node structure is invalid
        """
        if self.is_leaf_node():
            if self.value is None:
                raise ValueError("Leaf node must have a value")
            return self.value
        
        if self.feature is None or self.threshold is None:
            raise ValueError("Internal node must have feature and threshold")
        
        if X[self.feature] <= self.threshold:
            if self.left is None:
                raise ValueError("Internal node must have left child")
            return self.left.predict(X)
        else:
            if self.right is None:
                raise ValueError("Internal node must have right child")
            return self.right.predict(X)
    
    def get_depth(self) -> int:
        """
        Get the maximum depth of the subtree rooted at this node.
        
        Returns:
            Maximum depth of the subtree
        """
        if self.is_leaf_node():
            return self.depth
        
        left_depth = self.left.get_depth() if self.left else self.depth
        right_depth = self.right.get_depth() if self.right else self.depth
        
        return max(left_depth, right_depth)
    
    def get_n_leaves(self) -> int:
        """
        Get the number of leaf nodes in the subtree rooted at this node.
        
        Returns:
            Number of leaf nodes
        """
        if self.is_leaf_node():
            return 1
        
        left_leaves = self.left.get_n_leaves() if self.left else 0
        right_leaves = self.right.get_n_leaves() if self.right else 0
        
        return left_leaves + right_leaves
    
    def get_feature_importance(self, n_features: int) -> np.ndarray:
        """
        Calculate feature importance for the subtree rooted at this node.
        
        Args:
            n_features: Total number of features
            
        Returns:
            Array of feature importances
        """
        importance = np.zeros(n_features)
        
        if not self.is_leaf_node() and self.feature is not None:
            # Calculate importance as weighted decrease in MSE
            left_samples = self.left.n_samples if self.left else 0
            right_samples = self.right.n_samples if self.right else 0
            
            if self.n_samples > 0:
                left_mse = self.left.mse if self.left else 0
                right_mse = self.right.mse if self.right else 0
                
                weighted_mse = (
                    (left_samples * left_mse + right_samples * right_mse) 
                    / self.n_samples
                )
                
                importance[self.feature] = (
                    self.n_samples * (self.mse - weighted_mse)
                )
            
            # Recursively add importance from children
            if self.left:
                importance += self.left.get_feature_importance(n_features)
            if self.right:
                importance += self.right.get_feature_importance(n_features)
        
        return importance
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert node to dictionary representation for serialization.
        
        Returns:
            Dictionary representation of the node
        """
        node_dict = {
            'depth': self.depth,
            'n_samples': self.n_samples,
            'mse': self.mse,
            'is_leaf': self.is_leaf
        }
        
        if self.is_leaf_node():
            node_dict['value'] = self.value
        else:
            node_dict.update({
                'feature': self.feature,
                'threshold': self.threshold,
                'left': self.left.to_dict() if self.left else None,
                'right': self.right.to_dict() if self.right else None
            })
        
        return node_dict
    
    @classmethod
    def from_dict(cls, node_dict: Dict[str, Any]) -> 'Node':
        """
        Create node from dictionary representation.
        
        Args:
            node_dict: Dictionary representation of the node
            
        Returns:
            Node instance
        """
        if node_dict['is_leaf']:
            return cls(
                value=node_dict['value'],
                depth=node_dict['depth'],
                n_samples=node_dict['n_samples'],
                mse=node_dict['mse']
            )
        else:
            left = cls.from_dict(node_dict['left']) if node_dict['left'] else None
            right = cls.from_dict(node_dict['right']) if node_dict['right'] else None
            
            return cls(
                feature=node_dict['feature'],
                threshold=node_dict['threshold'],
                left=left,
                right=right,
                depth=node_dict['depth'],
                n_samples=node_dict['n_samples'],
                mse=node_dict['mse']
            )
    
    def __repr__(self) -> str:
        """
        String representation of the node.
        
        Returns:
            String representation
        """
        if self.is_leaf_node():
            return f"LeafNode(value={self.value:.4f}, n_samples={self.n_samples})"
        else:
            return (
                f"InternalNode(feature={self.feature}, threshold={self.threshold:.4f}, "
                f"n_samples={self.n_samples}, depth={self.depth})"
            )
    
    def print_tree(self, indent: str = "") -> None:
        """
        Print the tree structure starting from this node.
        
        Args:
            indent: Current indentation string
        """
        if self.is_leaf_node():
            print(f"{indent}Leaf: value={self.value:.4f}, samples={self.n_samples}")
        else:
            print(
                f"{indent}Node: feature={self.feature}, "
                f"threshold={self.threshold:.4f}, samples={self.n_samples}"
            )
            if self.left:
                print(f"{indent}├── Left:")
                self.left.print_tree(indent + "│   ")
            if self.right:
                print(f"{indent}└── Right:")
                self.right.print_tree(indent + "    ")