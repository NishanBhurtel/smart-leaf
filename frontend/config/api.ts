import { Platform } from 'react-native';
import Constants from 'expo-constants';

// Base IP configuration - change this to your machine's IP address
const BASE_IP = '10.10.0.118';
const BASE_PORT = '5000';

// Configure backend URL based on platform
export const API_BASE_URL = Platform.OS === 'android' && !Constants.isDevice
  ? `http://${BASE_IP}:${BASE_PORT}` // Android emulator (AVD)
  : `http://${BASE_IP}:${BASE_PORT}`;

// API endpoints
export const API_ENDPOINTS = {
  LOGIN: `${API_BASE_URL}/login`,
  SIGNUP: `${API_BASE_URL}/signup`,
  PREDICT: `${API_BASE_URL}/predict`,
  USERS: `${API_BASE_URL}/users`,
};
