package com.example.csv;

import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.HashMap;

public class CsvParser {
    private final char delimiter;

    public CsvParser(char delimiter) {
        this.delimiter = delimiter;
    }

    public List<Map<String, String>> parse(String headerLine, List<String> rows) {
        if (headerLine == null || rows == null) throw new IllegalArgumentException("input null");
        String[] headers = headerLine.split(String.valueOf(delimiter));
        List<Map<String, String>> table = new ArrayList<>();
        for (String row : rows) {
            String[] fields = row.split(String.valueOf(delimiter), -1);
            if (fields.length != headers.length)
                throw new IllegalArgumentException("row length mismatch");
            Map<String, String> map = new HashMap<>();
            for (int i = 0; i < headers.length; i++) {
                map.put(headers[i], fields[i]);
            }
            table.add(map);
        }
        return table;
    }
}
