const API = window.location.origin;

// ===== Questionnaire Submit =====
document.getElementById("questionnaire").onsubmit = async function(e) {
  e.preventDefault();

  const getRadio = name => document.querySelector(`input[name="${name}"]:checked`)?.value || "";
  const getNum = id => parseFloat(document.getElementById(id).value);

  const sectors = Array.from(document.querySelectorAll('input[name="sectors_to_avoid"]:checked'))
                        .map(cb => cb.value)
                        .filter(v => v !== "e");

  const data = {
    age: getNum("age"),
    income: getNum("income"),
    investment_goal: "retirement",
    risk_tolerance: getNum("risk_tolerance"),
    primary_goal: getRadio("primary_goal"),
    access_time: getRadio("access_time"),
    income_stability: getRadio("income_stability"),
    emergency_fund: getRadio("emergency_fund"),
    investment_plan: getRadio("investment_plan"),
    initial_investment: getRadio("initial_investment"),
    reaction_to_loss: getRadio("reaction_to_loss"),
    investing_experience: getRadio("investing_experience"),
    geographical_focus: getRadio("geographical_focus"),
    esg_preference: getRadio("esg_preference"),
    sectors_to_avoid: sectors
  };

  // Risk Assessment
  const res = await fetch(API + "/assess", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });
  const assessResult = await res.json();

  // ETF Recommendations
  const etfRes = await fetch(API + "/recommend/" + assessResult.risk_bucket);
  const etfResult = await etfRes.json();

  // Show Result
  const resultDiv = document.getElementById("result");
  resultDiv.innerHTML = `
    <strong>Risk Bucket:</strong> ${assessResult.risk_bucket}<br>
    <strong>Score:</strong> ${assessResult.score}<br><br>
    <strong>Target Volatility:</strong> ${assessResult.target_volatility_pct_range[0]}–${assessResult.target_volatility_pct_range[1]}%<br>
    <strong>Typical Allocation:</strong> ${assessResult.typical_allocation_hint.equity_pct}% Equity / ${assessResult.typical_allocation_hint.bond_pct}% Bond<br><br>
    <strong>ETF Recommendations (Cluster ${etfResult.cluster}):</strong><br><br>
  `;

  Object.entries(etfResult.etfs).forEach(([ticker, info], idx) => {
    if (!info.history) return;

    const container = document.createElement("div");
    container.innerHTML = `
      <h4>${ticker} — Return: ${(info.ann_return * 100).toFixed(2)}%, Volatility: ${(info.volatility * 100).toFixed(2)}%</h4>
      <canvas id="chart${idx}" height="120"></canvas>
      <hr>
    `;
    resultDiv.appendChild(container);

    new Chart(document.getElementById(`chart${idx}`), {
      type: 'line',
      data: {
        labels: Object.keys(info.history),
        datasets: [{
          label: 'Monthly Price (Last 3y)',
          data: Object.values(info.history),
          borderColor: 'steelblue',
          borderWidth: 2,
          fill: false,
          tension: 0.25
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false }, title: { display: false } },
        scales: { x: { ticks: { autoSkip: true, maxTicksLimit: 6 }, title: { display: true, text: "Time (Year-Month)" } },
                  y: { title: { display: true, text: "Price (€)" }, beginAtZero: false } }
      }
    });
  });
};

// ===== Chatbot =====
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");
const chatMessages = document.getElementById("chat-messages");

chatSend.onclick = async () => {
  const msg = chatInput.value.trim();
  if (!msg) return;

  appendMessage(msg, "chat-user");
  chatInput.value = "";

  const res = await fetch(API + "/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: msg })
  });
  const data = await res.json();
  appendMessage(data.reply, "chat-bot");
};

chatInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") chatSend.click();
});

function appendMessage(msg, cls) {
  const div = document.createElement("div");
  div.className = `chat-message ${cls}`;
  div.textContent = msg;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}
