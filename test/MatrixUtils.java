package com.example.math;

public class MatrixUtils {
    public static int[][] transpose(int[][] matrix) {
        if (matrix == null) throw new IllegalArgumentException("matrix null");
        int rows = matrix.length;
        int cols = matrix[0].length;
        int[][] result = new int[cols][rows];
        for (int i = 0; i < rows; i++) {
            if (matrix[i].length != cols) throw new IllegalArgumentException("ragged matrix");
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    public static int[][] multiply(int[][] a, int[][] b) {
        if (a == null || b == null) throw new IllegalArgumentException("matrix null");
        int aRows = a.length, aCols = a[0].length, bCols = b[0].length;
        if (b.length != aCols) throw new IllegalArgumentException("incompatible dims");
        int[][] result = new int[aRows][bCols];
        for (int i = 0; i < aRows; i++) {
            if (a[i].length != aCols) throw new IllegalArgumentException("ragged matrix");
            for (int j = 0; j < bCols; j++) {
                int sum = 0;
                for (int k = 0; k < aCols; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }
}
