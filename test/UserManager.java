package com.example.users;

import java.util.List;
import java.util.ArrayList;
import java.util.Optional;

public class UserManager {
    private final List<User> users = new ArrayList<>();

    public void addUser(User u) {
        if (u == null) throw new NullPointerException("user is null");
        users.add(u);
    }

    public Optional<User> findByUsername(String username) {
        return users.stream()
            .filter(u -> u.getUsername().equals(username))
            .findFirst();
    }

    public boolean removeUser(String username) {
        Optional<User> found = findByUsername(username);
        found.ifPresent(users::remove);
        return found.isPresent();
    }

    public static class User {
        private final String username;
        private final String email;

        public User(String username, String email) {
            if (username.isBlank() || email.isBlank())
                throw new IllegalArgumentException("username/email blank");
            this.username = username;
            this.email = email;
        }

        public String getUsername() {
            return username;
        }

        public String getEmail() {
            return email;
        }
    }
}
