import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { Keypair, SystemProgram, PublicKey } from "@solana/web3.js";
import { EmbeddingsStorage } from "../target/types/embeddings_storage";
import { sha256 } from "@noble/hashes/sha256";
import { assert } from "chai";

// Import the constants from the program if possible, or define them locally
const EMBEDDING_DIMENSION = 32; // Must match the program's constant
const MAX_URI_LENGTH = 64; // Must match the program's constant

// Helper to generate a random embedding
const generateRandomEmbedding = (): number[] => {
  return Array.from({ length: EMBEDDING_DIMENSION }, () => Math.random() * 2 - 1); // Values between -1 and 1
};

// Helper to calculate cosine similarity off-chain for verification
const calculateCosineSimilarity = (vec1: number[], vec2: number[]): number => {
    let dotProduct = 0;
    let magnitude1 = 0;
    let magnitude2 = 0;

    for (let i = 0; i < EMBEDDING_DIMENSION; i++) {
        dotProduct += vec1[i] * vec2[i];
        magnitude1 += vec1[i] * vec1[i];
        magnitude2 += vec2[i] * vec2[i];
    }

    magnitude1 = Math.sqrt(magnitude1);
    magnitude2 = Math.sqrt(magnitude2);

    if (magnitude1 === 0 || magnitude2 === 0) {
        return 0; // Or throw an error, depending on how you want to handle zero-magnitude vectors
    }

    return dotProduct / (magnitude1 * magnitude2);
};


describe("embeddings_onchain", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.EmbeddingsStorage as Program<EmbeddingsStorage>;

  const createdEmbeddings: { pda: PublicKey; uri: string; embedding: number[] }[] = [];

  it("Initializes multiple embedding accounts!", async () => {
    const numEmbeddings = 5;

    for (let i = 0; i < numEmbeddings; i++) {
      const uri = `https://example.com/data/${i}`;
      const embedding = generateRandomEmbedding();
      
      // DEBUGGING: Log the input URI
      console.log(`Attempting to initialize for URI: ${uri}`);

      const seedBuffer = Buffer.from(sha256(uri));
      
      // DEBUGGING: Log the generated seedBuffer and its hex representation
      console.log(`Seed Buffer for ${uri}:`, seedBuffer.toString('hex'));

      const [embeddingPda, _bump] = PublicKey.findProgramAddressSync(
        [Buffer.from("embedding"), seedBuffer],
        program.programId
      );

      // DEBUGGING: Log the derived PDA
      console.log(`Derived PDA for ${uri}: ${embeddingPda.toBase58()}`);

      // Check if this PDA has already been seen (this should be the core issue)
      if (createdEmbeddings.some(e => e.pda.equals(embeddingPda))) {
        console.warn(`WARNING: Attempting to initialize duplicate PDA: ${embeddingPda.toBase58()} for URI: ${uri}. Skipping.`);
        continue; // Skip trying to create the same account again
      }

      try {
        await program.methods
          .initializeEmbedding(
            // FIX: Convert Buffer to Array<number> when passing as instruction argument for Vec<u8>
            seedBuffer,
            embedding as number[],
            uri
          )
          .accounts({
            embeddingAccount: embeddingPda,
            signer: provider.wallet.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .rpc();

        const account = await program.account.embedding.fetch(embeddingPda);
        console.log(`Initialized embedding account for ${uri}:`, account.owner.toBase58());

        createdEmbeddings.push({ pda: embeddingPda, uri, embedding });
      } catch (error) {
        console.error(`Failed to initialize embedding for URI ${uri}:`, error);
        throw error;
      }
    }
    assert.equal(createdEmbeddings.length, numEmbeddings);
  });

  it("Performs on-chain similarity search", async () => {
    assert.isTrue(createdEmbeddings.length > 0, "No embeddings were initialized to search.");

    const queryEmbedding = createdEmbeddings[0].embedding;
    const queryUri = createdEmbeddings[0].uri;

    const accountsToSearch = createdEmbeddings.map(e => e.pda);

    console.log("Querying with embedding from URI:", queryUri);
    console.log("Accounts to search:", accountsToSearch.map(pk => pk.toBase58()));

    const tx = await program.methods
      .findSimilarEmbedding(queryEmbedding as number[])
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

    // --- Start of enhanced logging for log retrieval ---
    await new Promise(resolve => setTimeout(resolve, 500)); // Wait for 500ms

    const transaction = await provider.connection.getParsedTransaction(tx, { commitment: 'confirmed' });

    if (!transaction) {
      console.error(`ERROR: getParsedTransaction returned null for transaction ID: ${tx}. Transaction might not be confirmed or found.`);
      assert.fail(`Transaction ${tx} not found or confirmed.`);
    }

    if (transaction.meta?.err) {
      console.error(`ERROR: Transaction failed with error:`, transaction.meta.err);
      console.error(`Full transaction meta:`, JSON.stringify(transaction.meta, null, 2));
      assert.fail(`Transaction ${tx} failed with error: ${JSON.stringify(transaction.meta.err)}`);
    }

    const logs = transaction.meta?.logMessages;

    let foundBestMatchUri: string | null = null;
    let foundBestSimilarity: number | null = null;

    if (logs) {
      console.log("SUCCESS: Received logs for transaction:", logs); // Log all received logs
      for (const log of logs) {
        if (log.includes("Best match found:")) {
          const uriMatch = log.match(/URI = ([^,]+)/);
          const similarityMatch = log.match(/Similarity = ([\d.-]+)/);

          if (uriMatch && similarityMatch) {
              foundBestMatchUri = uriMatch[1];
              foundBestSimilarity = parseFloat(similarityMatch[1]);
          }
          break;
        }
      }
    } else {
      console.warn(`WARN: No log messages found in transaction meta for ${tx}.`);
      console.warn(`Transaction meta (if available):`, JSON.stringify(transaction.meta, null, 2));
    }
    // --- End of enhanced logging for log retrieval ---
    console.log("Found best match URI from logs:", foundBestMatchUri);
    console.log("Found best similarity from logs:", foundBestSimilarity);

    assert.isNotNull(foundBestMatchUri, "Should have found a best match URI in logs");
    assert.isNotNull(foundBestSimilarity, "Should have found a best similarity in logs");

    assert.equal(foundBestMatchUri, queryUri);
    assert.approximately(foundBestSimilarity!, 1.0, 1e-6);

    const newQueryEmbedding = generateRandomEmbedding();
    const newTx = await program.methods
      .findSimilarEmbedding(newQueryEmbedding as number[])
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

    // FIX: Use getParsedTransaction instead of getTransaction
    await new Promise(resolve => setTimeout(resolve, 500)); // Wait for 500ms
    
    const transaction2 = await provider.connection.getParsedTransaction(newTx, { commitment: 'confirmed' });
    const newLogs = transaction2?.meta?.logMessages;

    let newFoundBestMatchUri: string | null = null;
    let newFoundBestSimilarity: number | null = null;

    if (newLogs) {
      for (const log of newLogs) {
        if (log.includes("Best match found:")) {
          const uriMatch = log.match(/URI = ([^,]+)/);
          const similarityMatch = log.match(/Similarity = ([\d.-]+)/);

          if (uriMatch && similarityMatch) {
              newFoundBestMatchUri = uriMatch[1];
              newFoundBestSimilarity = parseFloat(similarityMatch[1]);
          }
          break;
        }
      }
    }

    console.log("New Query - Found best match URI from logs:", newFoundBestMatchUri);
    console.log("New Query - Found best similarity from logs:", newFoundBestSimilarity);

    let expectedBestSimilarity = -2.0;
    let expectedBestUri = "";
    for (const emb of createdEmbeddings) {
        const sim = calculateCosineSimilarity(newQueryEmbedding, emb.embedding);
        if (sim > expectedBestSimilarity) {
            expectedBestSimilarity = sim;
            expectedBestUri = emb.uri;
        }
    }

    assert.isNotNull(newFoundBestMatchUri, "Should have found a best match URI for new query");
    assert.isNotNull(newFoundBestSimilarity, "Should have found a best similarity for new query");
    assert.equal(newFoundBestMatchUri, expectedBestUri, "On-chain best match should match off-chain best match");
    assert.approximately(newFoundBestSimilarity!, expectedBestSimilarity, 1e-6, "On-chain similarity should be approximately equal to off-chain");

  });
});