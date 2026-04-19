const API = "http://127.0.0.1:5000";

let pieChart, barChart;

// 🔹 Get Analysis + Load Charts
function getAnalysis() {
    fetch(API + "/analyze")
        .then(res => res.json())
        .then(data => {
            document.getElementById("result").innerHTML =
                `<b>Total Sales:</b> ${data.total_sales}<br>
                 <b>Total Profit:</b> ${data.total_profit}<br>
                 <b>Avg Sales:</b> ${data.avg_sales}`;

            loadCharts();
        })
        .catch(err => console.error(err));
}

// 🔹 Load Charts
function loadCharts() {
    fetch(API + "/charts")
        .then(res => res.json())
        .then(data => {

            // Destroy old charts
            if (pieChart) pieChart.destroy();
            if (barChart) barChart.destroy();

            // 🥧 PIE CHART
            const pieCtx = document.getElementById("pieChart").getContext("2d");

            pieChart = new Chart(pieCtx, {
                type: "pie",
                data: {
                    labels: Object.keys(data.category_sales),
                    datasets: [{
                        data: Object.values(data.category_sales),
                        backgroundColor: [
                            "#4bc0c0",
                            "#ff6384",
                            "#ffcd56"
                        ]
                    }]
                },
                options: {
                    plugins: {
                        legend: {
                            labels: {
                                color: "white",
                                font: { size: 14 }
                            }
                        },
                        tooltip: {
                            bodyColor: "white",
                            titleColor: "white"
                        }
                    }
                }
            });

            // 📊 BAR CHART
            const barCtx = document.getElementById("barChart").getContext("2d");

            barChart = new Chart(barCtx, {
                type: "bar",
                data: {
                    labels: Object.keys(data.region_profit),
                    datasets: [{
                        label: "Profit",
                        data: Object.values(data.region_profit),
                        backgroundColor: "#36a2eb"
                    }]
                },
                options: {
                    plugins: {
                        legend: {
                            labels: {
                                color: "white"
                            }
                        },
                        tooltip: {
                            bodyColor: "white",
                            titleColor: "white"
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                color: "white"
                            },
                            title: {
                                display: true,
                                text: "Region",
                                color: "white"
                            }
                        },
                        y: {
                            ticks: {
                                color: "white"
                            },
                            title: {
                                display: true,
                                text: "Profit",
                                color: "white"
                            }
                        }
                    }
                }
            });

        })
        .catch(err => console.error(err));
}

// 🔹 Predict Profit
function predict() {
    const sales = document.getElementById("salesInput").value;

    fetch(API + "/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sales: sales })
    })
        .then(res => res.json())
        .then(data => {
            document.getElementById("prediction").innerHTML =
                `<b>Predicted Profit:</b> ${data.predicted_profit}`;
        })
        .catch(err => console.error(err));
}