
# External Services

resource "aws_iam_user" "external_secrets" {
  name = "external-secrets"
}

resource "aws_iam_access_key" "external_secrets" {
  user = aws_iam_user.external_secrets.name
}

resource "aws_iam_role" "external_secrets" {
  name = "external-secrets"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          AWS = aws_iam_user.external_secrets.arn
        },
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ssm_read_only_access" {
  role       = aws_iam_role.external_secrets.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMReadOnlyAccess"
}

resource "aws_iam_policy" "user_assume_role_policy" {
  name = "UserAssumeExternalSecretsRole"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = "sts:AssumeRole",
        Resource = aws_iam_role.external_secrets.arn
      }
    ]
  })
}

resource "aws_iam_user_policy_attachment" "assume_role" {
  user       = aws_iam_user.external_secrets.name
  policy_arn = aws_iam_policy.user_assume_role_policy.arn
}


# Backend service

resource "aws_iam_user" "backend_service" {
  name = "backend-service"
}

resource "aws_iam_access_key" "backend_service" {
  depends_on = [
    aws_iam_user.backend_service
  ]
  user = aws_iam_user.backend_service.name
}


resource "aws_iam_policy" "s3_rw_policy" {
  depends_on = [
    aws_s3_bucket.develop,
    aws_s3_bucket.main,
    aws_s3_bucket.datalake,
    aws_s3_bucket.labelstudio,
  ]
  name        = "S3ReadWritePolicy"
  description = "Policy that grants read-write access to S3"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
        ],
        Effect = "Allow",
        Resource = [
          "${aws_s3_bucket.uploads.arn}/*",
          aws_s3_bucket.uploads.arn,
          "${aws_s3_bucket.reports.arn}/*",
          aws_s3_bucket.reports.arn,
          "${aws_s3_bucket.develop.arn}/*",
          aws_s3_bucket.develop.arn,
          "${aws_s3_bucket.main.arn}/*",
          aws_s3_bucket.main.arn,
          "${aws_s3_bucket.datalake.arn}/*",
          aws_s3_bucket.datalake.arn,
          "${aws_s3_bucket.labelstudio.arn}/*",
          aws_s3_bucket.labelstudio.arn,
        ]
      },
    ]
  })
}

resource "aws_iam_user_policy_attachment" "role_policy_attach" {
  depends_on = [
    aws_iam_user.backend_service,
    aws_iam_policy.s3_rw_policy
  ]
  user       = aws_iam_user.backend_service.name
  policy_arn = aws_iam_policy.s3_rw_policy.arn
}

# LabelStudio

resource "aws_iam_user" "label_studio" {
  name = "label-studio"
}

resource "aws_iam_access_key" "label_studio" {
  depends_on = [
    aws_iam_user.label_studio
  ]
  user = aws_iam_user.label_studio.name
}

resource "aws_iam_policy" "label_studio_s3_rw_policy" {
  depends_on = [
    aws_s3_bucket.labelstudio,
  ]
  name        = "LabelStudio_S3ReadWritePolicy"
  description = "Policy that grants read-write access to LabelStudio S3 bucket"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
        ],
        Effect = "Allow",
        Resource = [
          "${aws_s3_bucket.labelstudio.arn}/*",
          aws_s3_bucket.labelstudio.arn,
        ]
      },
    ]
  })
}

resource "aws_iam_user_policy_attachment" "label_studio_role_policy_attach" {
  depends_on = [
    aws_iam_user.label_studio,
    aws_iam_policy.label_studio_s3_rw_policy
  ]
  user       = aws_iam_user.label_studio.name
  policy_arn = aws_iam_policy.label_studio_s3_rw_policy.arn
}

# EBS CSI driver

resource "aws_iam_user" "ebs_csi_driver" {
  name = "ebs-csi-driver"
}

resource "aws_iam_access_key" "ebs_csi_driver" {
  depends_on = [
    aws_iam_user.ebs_csi_driver
  ]
  user = aws_iam_user.ebs_csi_driver.name
}

resource "aws_iam_user_policy_attachment" "ebs_csi_driver_role_policy_attach" {
  depends_on = [
    aws_iam_user.ebs_csi_driver
  ]
  user       = aws_iam_user.ebs_csi_driver.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
}
