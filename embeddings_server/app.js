const express = require("express");
const anchor = require("@coral-xyz/anchor");
const { Program } = require("@coral-xyz/anchor");
const { SystemProgram, PublicKey, Connection, Keypair, SendTransactionError } = require("@solana/web3.js");
const { sha256 } = require("@noble/hashes/sha256");
const ProgramIDL = require("./target/idl/embeddings_onchain.json");
const {
  airdropIfRequired
} = require("@solana-developers/helpers");
const fs = require('fs');

const app = express();
const connection = new Connection("http://host.docker.internal:8899");
const wallet = new anchor.Wallet(new Keypair());

const provider = new anchor.AnchorProvider(connection, wallet, {
  commitment: "processed",
});
anchor.setProvider(provider);
const programId = ProgramIDL.metadata.address;
const program = new Program(
  ProgramIDL,
  provider,
  programId,
);
let createdEmbeddings = [];

async function saveEmbeddings() {
  fs.writeFileSync('./createdEmbeddings.json', JSON.stringify(createdEmbeddings, null, 2), 'utf-8');
  console.log('Saved createdEmbeddings to createdEmbeddings.json');
}

function loadEmbeddings() {
  createdEmbeddings = JSON.parse(fs.readFileSync('./createdEmbeddings.json', 'utf-8'));
  console.log('Loaded from createdEmbeddings.json:', createdEmbeddings);
}

async function main() {
  const LAMPORTS_PER_SOL = 1000000000;
  await airdropIfRequired(
    connection, 
    wallet.publicKey, 
  0.5 * LAMPORTS_PER_SOL,
  1 * LAMPORTS_PER_SOL,);

  console.log(wallet.publicKey);

  console.log(wallet);

  app.set("trust proxy", true);
  app.use(express.json()); // for parsing application/json

  app.post("/embeddings", postEmbeddingsHandler);
  app.get("/embeddings", queryEmbeddingsHandler);
    
}

loadEmbeddings()

main()

async function postEmbeddingsHandler(req, res) {
  try {
    const { embeddings: embeddingsInput, ids } = req.body;

    if (!Array.isArray(embeddingsInput) || !Array.isArray(ids)) {
      return res.status(400).json({ error: "'embeddings' and 'ids' must be arrays in request body" });
    }
    if (embeddingsInput.length !== ids.length) {
      return res.status(400).json({ error: "'embeddings' and 'ids' arrays must have the same length" });
    }

    for (const embedding of embeddingsInput) {
      if (!Array.isArray(embedding) || !embedding.every(n => typeof n === 'number')) {
        return res.status(400).json({ error: "Each embedding must be an array of numbers" });
      }
    }
    console.log(embeddingsInput);
    const embeddings = embeddingsInput;
    const result = await addOnchain(embeddings, ids);
    res.json({
      "status": "success",
      "result": result
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({ error: 'Internal server error', details: error.message || error });
  }
}

async function queryEmbeddingsHandler(req, res) {
  try {
    const embeddingInput = JSON.parse(req.query.embeddings);
    if (!Array.isArray(embeddingInput) || !embeddingInput.every(n => typeof n === 'number')) {
      console.log(req.query)
      res.status(400).json({ error: "Missing or invalid 'embedding' array in request query" });
      return;
    }
    const embedding = numberArrayToFloat32Array(embeddingInput);
    const result = await queryOnchain(embedding);
    res.json({
      "ids": [result ? parseInt(result): 1] // default value in case of null
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({ error: "Internal server error", details: error.message || error });
  }
}

function numberArrayToFloat32Array(arr) {
  return new Float32Array(arr);
}

async function queryOnchain(embeddings) {
  console.log("query received: ", embeddings)
  if (createdEmbeddings.length === 0) {
    throw new Error("No embeddings available for query.");
  }
  const queryEmbedding = embeddings;
  const accountsToSearch = createdEmbeddings.map(e => e.pda);

  console.log("Querying on-chain with provided embedding");
  console.log("Accounts to search:", accountsToSearch.map(pk => pk.toBase58()));

  const tx = await program.methods
    .findSimilarEmbedding(Array.from(queryEmbedding))
    .accounts({
      signer: provider.wallet.publicKey,
      systemProgram: SystemProgram.programId,
    })
    .remainingAccounts(
      accountsToSearch.map(pubkey => ({
        pubkey,
        isWritable: false,
        isSigner: false,
      }))
    )
    .rpc();

  console.log("Transaction ID for search:", tx);

  await new Promise(resolve => setTimeout(resolve, 500)); // Wait for 500ms

  const transaction = await provider.connection.getParsedTransaction(tx, { commitment: 'confirmed' });

  if (!transaction) {
    console.error(`ERROR: getParsedTransaction returned null for transaction ID: ${tx}. Transaction might not be confirmed or found.`);
  }

  if (transaction.meta?.err) {
    console.error("ERROR: Transaction failed with error:", transaction.meta.err);
    console.error("Full transaction meta:", JSON.stringify(transaction.meta, null, 2));
  }

  const logs = transaction.meta?.logMessages;
  let foundBestMatchUri = null;
  let foundBestSimilarity = null;

  if (logs) {
    console.log("Received logs for transaction:", logs);
    for (const log of logs) {
      if (log.includes("Best match found:")) {
        const uriMatch = log.match(/URI = (document\/\d+)/);;

        if (uriMatch) {
          foundBestMatchUri = uriMatch[1];
        }
        break;
      }
    }
  } else {
    console.warn(`No log messages found in transaction meta for ${tx}.`);
    console.warn("Transaction meta (if available):", JSON.stringify(transaction.meta, null, 2));
  }

  console.log("Found best match URI:", foundBestMatchUri);

  return foundBestMatchUri ? foundBestMatchUri.match(/document\/(.+)/)[1] : null;
}

async function addOnchain(embeddings, ids) {
  const numEmbeddings = embeddings.length;
  const results = [];
  for (let i = 0; i < numEmbeddings; i++) {


    const uri = `document/${ids[i]}`;

    
    if(createdEmbeddings.find((val, ind) => val.uri == uri)) {
      results.push({
        "id": ids[i],
        "status": "Id is used onchain. Embedding was not updated."
      });
      continue;
    }

    const embedding = embeddings[i];
    console.log("uri: ", uri)
    console.log("embeddings: ", embedding)

    console.log(`Attempting to initialize for URI: ${uri}`);
    const seedBuffer = Buffer.from(sha256(uri));
    console.log(`Seed Buffer for ${uri}:`, seedBuffer.toString("hex"));

    const [embeddingPda, _bump] = PublicKey.findProgramAddressSync(
      [Buffer.from("embedding"), seedBuffer],
      program.programId
    );

    console.log(`Derived PDA for ${uri}:`, embeddingPda.toBase58());

    if (createdEmbeddings.some(e => e.pda == embeddingPda)) {
      console.warn(`Duplicate PDA: ${embeddingPda.toBase58()} for URI: ${uri}. Skipping.`);
      continue;
    }

    try {
      await program.methods
        .initializeEmbedding(seedBuffer, embedding, uri)
        .accounts({
          embeddingAccount: embeddingPda,
          signer: provider.wallet.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .rpc();
      let account;
      for(let i = 1; i < 20; i ++) {
        await new Promise(resolve => setTimeout(resolve, 1000)); // Wait for 500ms
        console.log("waited for ", i, " seconds")
        try{
          account = await program.account.embedding.fetch(embeddingPda);
          break;
        }catch(e){
          console.log("couldn't get account after ", i ," seconds ")
        }
      }

      console.log(`Initialized embedding account for ${uri}:`, account.owner.toBase58());

      createdEmbeddings.push({ pda: embeddingPda, uri, embedding });
      results.push({
        "id": ids[i],
        "status": "Embedding added onchain."
      });
    } catch (error) {
      console.error(`Failed to initialize embedding for URI ${uri}:`, error);
      if (err instanceof SendTransactionError)
        console.log(err.getLogs())
      results.push({
        "id": ids[i],
        "status": "Embedding was not added onchain. Error: " + err
      });
    }
  }
  saveEmbeddings()
  console.log("Saved embed accounts: ", createdEmbeddings)
  return results;
}

module.exports = app;

