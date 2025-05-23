use anchor_lang::prelude::*;

// This is your program's public key and it will update
// automatically when you build the project.
declare_id!("A1g3ToDasaBD2Xc11rhALercNtNeaWmTS1VxCqhQ6Daa"); // REPLACE WITH YOUR PROGRAM ID

pub const EMBEDDING_DIMENSION: usize = 32; // Significantly reduced for on-chain search feasibility
pub const MAX_URI_LENGTH: usize = 64; // Adjusted for smaller accounts

#[program]
pub mod embeddings_onchain { // Ensure this matches your project name in Anchor.toml
    use super::*;

    pub fn initialize_embedding(
        ctx: Context<InitializeEmbedding>,
        _seed_data: Vec<u8>, // Prefixed with _ to suppress unused variable warning
        embedding_data: [f32; EMBEDDING_DIMENSION],
        uri: String,
    ) -> Result<()> {
        let embedding_account = &mut ctx.accounts.embedding_account;

        if uri.len() > MAX_URI_LENGTH {
            return Err(ErrorCode::UriTooLong.into());
        }

        embedding_account.embedding = embedding_data;
        embedding_account.uri = uri;
        embedding_account.owner = ctx.accounts.signer.key();

        msg!("Initialized embedding account: {:?}", embedding_account.key());
        Ok(())
    }

    pub fn update_embedding(
        ctx: Context<UpdateEmbedding>,
        embedding_data: [f32; EMBEDDING_DIMENSION],
        uri: String,
    ) -> Result<()> {
        let embedding_account = &mut ctx.accounts.embedding_account;

        require_eq!(ctx.accounts.signer.key(), embedding_account.owner, ErrorCode::Unauthorized);

        if uri.len() > MAX_URI_LENGTH {
            return Err(ErrorCode::UriTooLong.into());
        }

        embedding_account.embedding = embedding_data;
        embedding_account.uri = uri;

        msg!("Updated embedding account: {:?}", embedding_account.key());
        Ok(())
    }

    pub fn find_similar_embedding(
        ctx: Context<FindSimilarEmbedding>,
        query_embedding: [f32; EMBEDDING_DIMENSION],
    ) -> Result<()> {
        let mut best_similarity: f32 = -1.0; // Cosine similarity ranges from -1 to 1
        let mut best_match_uri: String = String::new();
        let mut best_match_pubkey: Pubkey = Pubkey::default();

        msg!("Starting similarity search...");

        // Iterating through the provided accounts
        // We need to work with the data directly from AccountInfo to avoid lifetime issues
        for account_info in ctx.remaining_accounts.iter() {
            // Check if the account is owned by the program and deserialize it
            if account_info.owner != ctx.program_id {
                msg!("Skipping account {:?}: not owned by program", account_info.key());
                continue;
            }

            // FIX: Directly deserialize into Embedding struct
            // This copies the data and avoids the lifetime dependency on `account_info` itself.
            let embedding_account: Embedding = Embedding::try_deserialize(&mut &account_info.data.borrow()[..])?;

            // Calculate cosine similarity
            let similarity = calculate_cosine_similarity(
                &query_embedding,
                &embedding_account.embedding,
            )?;

            msg!(
                "Comparing with URI: {}, Pubkey: {:?}, Similarity: {}",
                embedding_account.uri,
                account_info.key(), // Use account_info.key() directly for the Pubkey
                similarity
            );

            if similarity > best_similarity {
                best_similarity = similarity;
                best_match_uri = embedding_account.uri.clone();
                best_match_pubkey = account_info.key(); // Use account_info.key() directly
            }
        }

        if best_similarity == -1.0 { // No embeddings found or processed
             msg!("No suitable embeddings found for comparison.");
        } else {
            msg!(
                "Best match found: URI = {}, Pubkey = {:?}, Similarity = {}",
                best_match_uri,
                best_match_pubkey,
                best_similarity
            );
        }

        Ok(())
    }
}

#[derive(Accounts)]
#[instruction(seed_data: Vec<u8>)]
pub struct InitializeEmbedding<'info> {
    #[account(
        init,
        payer = signer,
        space = 8 + Embedding::INIT_SPACE, // INIT_SPACE accounts for #[max_len]
        seeds = ["embedding".as_bytes(), &seed_data],
        bump
    )]
    pub embedding_account: Account<'info, Embedding>,
    #[account(mut)]
    pub signer: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct UpdateEmbedding<'info> {
    #[account(mut)]
    pub embedding_account: Account<'info, Embedding>,
    #[account(mut)]
    pub signer: Signer<'info>,
}

#[derive(Accounts)]
pub struct FindSimilarEmbedding<'info> {
    #[account(mut)]
    pub signer: Signer<'info>,
    pub system_program: Program<'info, System>,
    // `remaining_accounts` are handled by Anchor, no need to list them here.
}

#[account]
#[derive(Default, InitSpace)]
pub struct Embedding {
    pub embedding: [f32; EMBEDDING_DIMENSION],
    #[max_len(MAX_URI_LENGTH)]
    pub uri: String,
    pub owner: Pubkey,
}

// --- Helper function for cosine similarity ---
fn calculate_cosine_similarity(
    vec1: &[f32; EMBEDDING_DIMENSION],
    vec2: &[f32; EMBEDDING_DIMENSION],
) -> Result<f32> {
    let mut dot_product: f32 = 0.0;
    let mut magnitude1: f32 = 0.0;
    let mut magnitude2: f32 = 0.0;

    for i in 0..EMBEDDING_DIMENSION {
        dot_product += vec1[i] * vec2[i];
        magnitude1 += vec1[i] * vec1[i];
        magnitude2 += vec2[i] * vec2[i];
    }

    magnitude1 = magnitude1.sqrt();
    magnitude2 = magnitude2.sqrt();

    if magnitude1 == 0.0 || magnitude2 == 0.0 {
        return Err(ErrorCode::ZeroMagnitudeVector.into());
    }

    Ok(dot_product / (magnitude1 * magnitude2))
}


// Custom error codes
#[error_code]
pub enum ErrorCode {
    #[msg("Provided URI is too long.")]
    UriTooLong,
    #[msg("Unauthorized: Only the account owner can perform this action.")]
    Unauthorized,
    #[msg("Vector has zero magnitude, cannot calculate cosine similarity.")]
    ZeroMagnitudeVector,
}