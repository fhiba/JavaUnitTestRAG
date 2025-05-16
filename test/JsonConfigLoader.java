package com.example.config;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;

public class JsonConfigLoader {
    private final ObjectMapper mapper = new ObjectMapper();

    /**
     * Loads a JSON configuration file into a map.
     *
     * @param configPath the path to the JSON file
     * @return the parsed key/value map
     * @throws IOException if the file can’t be read or parsed
     */
    public Map<String, Object> loadConfig(Path configPath) throws IOException {
        if (configPath == null) {
            throw new IllegalArgumentException("configPath is null");
        }
        if (!Files.exists(configPath) || !Files.isRegularFile(configPath)) {
            throw new IOException("Config file not found: " + configPath);
        }

        byte[] bytes = Files.readAllBytes(configPath);
        return mapper.readValue(bytes, new TypeReference<Map<String, Object>>() {});
    }

    /**
     * Merges two configurations, with values in override taking precedence.
     */
    public Map<String, Object> mergeConfigs(
        Map<String, Object> base,
        Map<String, Object> override
    ) {
        if (base == null || override == null) {
            throw new IllegalArgumentException("Maps must not be null");
        }
        for (Map.Entry<String, Object> entry : override.entrySet()) {
            base.put(entry.getKey(), entry.getValue());
        }
        return base;
    }
}
