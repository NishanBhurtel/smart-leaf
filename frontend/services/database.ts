/**
 * Database service - DEPRECATED
 * 
 * This file is kept for reference only.
 * All authentication and user management is now handled by the backend API.
 * The users table is created and managed in the PostgreSQL database on the backend.
 * 
 * See:
 * - Backend: backend/app.py (PostgreSQL with user table)
 * - Frontend: contexts/AuthContext.tsx (API calls to backend)
 */

export interface User {
  id: number;
  name: string;
  email: string;
  password: string;
}

// This service is no longer used
// All database operations are now handled by the backend API
class DatabaseService {
  // Deprecated - do not use
}

export const databaseService = new DatabaseService();
