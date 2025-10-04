-- Fix Email Data - Remove invalid parent_id references
USE bitnami_suitecrm;

-- Update email records to remove invalid parent_id references
UPDATE emails 
SET parent_type = NULL, 
    parent_id = NULL 
WHERE id LIKE 'email-%';

-- Also update the assigned_user_id to ensure it's valid
UPDATE emails 
SET assigned_user_id = '1' 
WHERE id LIKE 'email-%' AND assigned_user_id IS NULL;

-- Display updated data
SELECT 'Updated Email Records:' as info;
SELECT id, name, type, status, assigned_user_id, parent_type, parent_id 
FROM emails 
WHERE id LIKE 'email-%';

-- Check if emails are now visible
SELECT 'Email count by status:' as info;
SELECT status, COUNT(*) as count 
FROM emails 
WHERE deleted = 0 
GROUP BY status;

