const form = document.getElementById("predict-form");
const resultDiv = document.getElementById("result");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const transactionData = {
    TransactionAmt: parseFloat(document.getElementById("amount").value),
    ProductCD: document.getElementById("product").value,
  };
  const modelChoice = document.getElementById("model-choice").value;

  try {
    const result = await predictTransaction(transactionData, modelChoice);

    resultDiv.innerHTML = `
      <p>Fraud Probability: ${(result.fraud_probability * 100).toFixed(2)}%</p>
      <p>Risk Level: ${result.risk_bucket}</p>
      <p>Model: ${result.model_version}</p>
    `;
    resultDiv.hidden = false;
  } catch (err) {
    resultDiv.innerHTML = `<p>Error: ${err.message}</p>`;
    resultDiv.hidden = false;
  }
});