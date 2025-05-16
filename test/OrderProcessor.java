package com.example.ragtest;

import java.util.*;

/**
 * A moderately complex class to manage orders and demonstrate business logic.
 */
public class OrderProcessor {
    private final List<Order> orders;
    private final Map<OrderStatus, List<Order>> ordersByStatus;

    public OrderProcessor() {
        orders = new ArrayList<>();
        ordersByStatus = new EnumMap<>(OrderStatus.class);
        for (OrderStatus status : OrderStatus.values()) {
            ordersByStatus.put(status, new ArrayList<>());
        }
    }

    /**
     * Adds a new order to the processor.
     */
    public void addOrder(Order order) {
        orders.add(order);
        ordersByStatus.get(order.getStatus()).add(order);
    }

    /**
     * Calculates the total price of all orders.
     */
    public double calculateTotal() {
        double total = 0;
        for (Order order : orders) {
            total += order.getQuantity() * order.getPricePerUnit();
        }
        return total;
    }

    /**
     * Calculates total price after applying a discount rate.
     * @throws IllegalArgumentException if discountRate is not between 0 and 1
     */
    public double calculateTotalWithDiscount(double discountRate) {
        if (discountRate < 0 || discountRate > 1) {
            throw new IllegalArgumentException("Discount rate must be between 0 and 1");
        }
        return calculateTotal() * (1 - discountRate);
    }

    /**
     * Counts the total quantities ordered per item name.
     */
    public Map<String, Long> countItems() {
        Map<String, Long> itemCounts = new HashMap<>();
        for (Order order : orders) {
            itemCounts.merge(order.getItemName(), (long) order.getQuantity(), Long::sum);
        }
        return itemCounts;
    }

    /**
     * Retrieves orders by their status.
     */
    public List<Order> getOrdersByStatus(OrderStatus status) {
        return Collections.unmodifiableList(
            ordersByStatus.getOrDefault(status, Collections.emptyList())
        );
    }

    /**
     * Possible statuses for an order.
     */
    public enum OrderStatus {
        PENDING, PROCESSING, SHIPPED, DELIVERED, CANCELLED
    }

    /**
     * Represents a single order with item, quantity, price, and status.
     */
    public static class Order {
        private final String itemName;
        private final int quantity;
        private final double pricePerUnit;
        private OrderStatus status;

        public Order(String itemName, int quantity, double pricePerUnit, OrderStatus status) {
            this.itemName = itemName;
            this.quantity = quantity;
            this.pricePerUnit = pricePerUnit;
            this.status = status;
        }

        public String getItemName() { return itemName; }
        public int getQuantity() { return quantity; }
        public double getPricePerUnit() { return pricePerUnit; }
        public OrderStatus getStatus() { return status; }
        public void setStatus(OrderStatus status) { this.status = status; }

        @Override
        public String toString() {
            return String.format("Order[item=%s, qty=%d, price=%.2f, status=%s]",
                                  itemName, quantity, pricePerUnit, status);
        }
    }
}
