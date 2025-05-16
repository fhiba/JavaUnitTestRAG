package com.example.rate;

import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicInteger;

public class TokenBucketRateLimiter {
    private final int capacity;
    private final Duration refillInterval;
    private AtomicInteger tokens;
    private Instant lastRefill;

    public TokenBucketRateLimiter(int capacity, Duration refillInterval) {
        if (capacity <= 0 || refillInterval.isZero() || refillInterval.isNegative()) {
            throw new IllegalArgumentException("Invalid capacity or interval");
        }
        this.capacity = capacity;
        this.refillInterval = refillInterval;
        this.tokens = new AtomicInteger(capacity);
        this.lastRefill = Instant.now();
    }

    public synchronized boolean tryConsume(int numTokens) {
        refillIfNeeded();
        if (numTokens <= 0) {
            throw new IllegalArgumentException("Requested tokens must be > 0");
        }
        if (tokens.get() >= numTokens) {
            tokens.addAndGet(-numTokens);
            return true;
        }
        return false;
    }

    private void refillIfNeeded() {
        Instant now = Instant.now();
        long intervals = Duration.between(lastRefill, now).dividedBy(refillInterval);
        if (intervals > 0) {
            int refillAmount = (int) Math.min(intervals * capacity, capacity - tokens.get());
            tokens.set(Math.min(capacity, tokens.get() + refillAmount));
            lastRefill = lastRefill.plus(refillInterval.multipliedBy(intervals));
        }
    }

    public int getAvailableTokens() {
        refillIfNeeded();
        return tokens.get();
    }
}
