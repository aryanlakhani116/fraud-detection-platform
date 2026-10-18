const API_BASE_URL = "http://127.0.0.1:8000";

async function predictTransaction(transactionData, modelChoice) {
  const response = await fetch(
    `${API_BASE_URL}/predict/transaction/?model=${modelChoice}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(transactionData),
    }
  );

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  return response.json();
}