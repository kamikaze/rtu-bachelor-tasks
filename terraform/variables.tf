variable "organization" {
  type     = string
  default  = "Bixority"
  nullable = false

  validation {
    condition     = length(var.organization) >= 3
    error_message = "Organization name must be at least 3 characters long."
  }
}

variable "project" {
  type     = string
  default  = "rtu-bachelor-tasks"
  nullable = false

  validation {
    condition     = length(var.project) >= 3
    error_message = "Project name must be at least 3 characters long."
  }
}

# AWS

variable "admin_ssh_public_key_name" {
  type     = string
  default  = "admin-ssh-key"
  nullable = false

  validation {
    condition     = length(var.admin_ssh_public_key_name) >= 3
    error_message = "Admin ssh key name must be at least 3 characters long."
  }
}

variable "admin_ssh_public_key" {
  type     = string
  default  = ""
  nullable = false
}

variable "aws_region" {
  type     = string
  default  = "eu-north-1"
  nullable = false
}

variable "s3_enabled" {
  type    = bool
  default = true
}

variable "ecr_enabled" {
  type    = bool
  default = true
}

variable "ec2_enabled" {
  type    = bool
  default = false
}

variable "rds_enabled" {
  type    = bool
  default = true
}

variable "db_username" {
  type        = string
  nullable    = false
  default     = "dev"
  description = "Username for a postgres database"
}

variable "db_password" {
  type        = string
  nullable    = false
  sensitive   = true
  description = "Password for a postgres database"

  validation {
    condition     = length(var.db_password) >= 18
    error_message = "Database password must be at least 18 characters long."
  }
}

variable "dataset_db_username" {
  type        = string
  nullable    = false
  default     = "dev"
  description = "Username for a dataset postgres database"
}

variable "dataset_db_password" {
  type        = string
  nullable    = false
  sensitive   = true
  description = "Password for a dataset postgres database"

  validation {
    condition     = length(var.dataset_db_password) >= 18
    error_message = "Database password must be at least 18 characters long."
  }
}


variable "dns_api_token" {
  type        = string
  nullable    = false
  sensitive   = true
  description = "DNS API token"
}

variable "dns_api_secret" {
  type        = string
  nullable    = false
  sensitive   = true
  description = "DNS API secret"
}

variable "eks_enabled" {
  type    = bool
  default = true
}

variable "eks_version" {
  type     = string
  default  = "1.32"
  nullable = false
}

variable "eks_cluster_name" {
  type     = string
  default  = "eks-dev"
  nullable = false
}

variable "eks_node_instance_type" {
  type     = string
  default  = "t4g.medium"
  nullable = false
}

# GitHub

variable "github_token" {
  sensitive = true
  type      = string
}

variable "github_org" {
  type = string
}

variable "github_repository" {
  type    = string
  default = "infrastructure"
}

variable "git_infra_repo_url" {
  type     = string
  default  = ""
  nullable = false

  validation {
    condition     = length(var.git_infra_repo_url) >= 6
    error_message = "Git repository URL must be at least 6 characters long."
  }
}

# FluxCD

variable "fluxcd_enabled" {
  type    = bool
  default = true
}

variable "flux_git_infra_target_path" {
  type     = string
  default  = "fluxcd/clusters/dev"
  nullable = false

  validation {
    condition     = length(var.flux_git_infra_target_path) >= 3
    error_message = "Git target path must be at least 3 characters long."
  }
}

variable "flux_git_infra_branch" {
  type     = string
  default  = "main"
  nullable = false

  validation {
    condition     = length(var.flux_git_infra_branch) >= 1
    error_message = "Git branch name must be at least 1 characters long."
  }
}
